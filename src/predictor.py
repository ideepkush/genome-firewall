"""Module 02 — Predictor: will each antibiotic work?

One regularized logistic regression per antibiotic over binary AMR features, wrapped in
two layers the challenge requires:

  1. a deterministic target gate applied *before* the model, so a drug the organism is
     intrinsically resistant to is never reported as "likely to work";
  2. probability calibration fitted on a held-out calibration split, so the confidence
     number shown to a clinician means what it says.

L1 regularization is deliberate: it drives most coefficients to zero, leaving a short
list of determinants per drug that can be shown as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from .drug_database import Drug, class_features_for_drug, get_drug
from .utils.calibration import (
    LIKELY_TO_FAIL,
    LIKELY_TO_WORK,
    NO_CALL,
    ProbabilityCalibrator,
    apply_no_call,
)

def _l1_logistic(C: float) -> LogisticRegression:
    """L1-penalized logistic regression, across the sklearn 1.8 API change.

    scikit-learn 1.8 deprecated `penalty=` in favour of `l1_ratio=`. Selecting the
    right keyword here keeps the sparse-coefficient behaviour we rely on for evidence
    reporting on both old and new installs.
    """
    common = dict(C=C, class_weight="balanced", max_iter=2000, solver="liblinear")
    major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, **common)
    return LogisticRegression(penalty="l1", **common)


EVIDENCE_KNOWN_DETERMINANT = "known resistance determinant detected"
EVIDENCE_STATISTICAL = "statistical association only"
EVIDENCE_NO_SIGNAL = "no known resistance signal found"
EVIDENCE_INTRINSIC = "intrinsic resistance (deterministic gate)"


@dataclass
class Prediction:
    """One antibiotic's result for one genome, with its supporting evidence.

    `resistance_probability` is the calibrated probability that the drug fails, and is
    always meaningful. `confidence` is how strongly the stated decision is supported and
    is only defined when a decision was actually made — a no-call has no confidence,
    because there is no claim to be confident about.
    """

    drug: str
    decision: str
    resistance_probability: float
    confidence: float | None
    evidence_type: str
    supporting_features: list[str] = field(default_factory=list)
    gate_note: str = ""

    @property
    def is_called(self) -> bool:
        return self.decision != NO_CALL

    def to_dict(self) -> dict:
        return {
            "drug": self.drug,
            "decision": self.decision,
            "resistance_probability": round(self.resistance_probability, 3),
            "confidence": round(self.confidence, 3) if self.confidence is not None else "",
            "evidence_type": self.evidence_type,
            "supporting_features": ", ".join(self.supporting_features) or "—",
            "gate_note": self.gate_note,
        }


class DrugModel:
    """Calibrated resistance classifier for a single antibiotic."""

    def __init__(self, drug_name: str, C: float = 0.1, low: float = 0.35, high: float = 0.65):
        self.drug_name = drug_name
        self.drug: Drug | None = get_drug(drug_name)
        self.C = C
        self.low = low
        self.high = high

        self.model = _l1_logistic(C)
        self.calibrator = ProbabilityCalibrator(method="auto")
        self.feature_names_: list[str] = []
        self.is_fitted_ = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_cal: pd.DataFrame | None = None,
        y_cal: np.ndarray | None = None,
    ) -> "DrugModel":
        """Fit the classifier, then calibrate on a disjoint split.

        Falls back to calibrating on training predictions if no calibration split is
        given, which is better than nothing but will read optimistic — the metrics
        report should say which path was used.
        """
        self.feature_names_ = list(X_train.columns)
        self.model.fit(X_train.to_numpy(), np.asarray(y_train, dtype=int))

        if X_cal is not None and len(X_cal) > 0:
            cal_scores = self.model.predict_proba(X_cal[self.feature_names_].to_numpy())[:, 1]
            self.calibrator.fit(cal_scores, np.asarray(y_cal, dtype=int))
        else:
            train_scores = self.model.predict_proba(X_train.to_numpy())[:, 1]
            self.calibrator.fit(train_scores, np.asarray(y_train, dtype=int))

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability that this drug will fail (i.e. the isolate is resistant)."""
        self._check_fitted()
        raw = self.model.predict_proba(X[self.feature_names_].to_numpy())[:, 1]
        return self.calibrator.transform(raw)

    def selected_features(self) -> list[tuple[str, float]]:
        """Non-zero coefficients, largest magnitude first — the model's learned evidence."""
        self._check_fitted()
        coefs = self.model.coef_.ravel()
        pairs = [(n, float(c)) for n, c in zip(self.feature_names_, coefs) if abs(c) > 1e-8]
        return sorted(pairs, key=lambda p: abs(p[1]), reverse=True)

    def explain(self, row: pd.Series) -> tuple[str, list[str]]:
        """Classify the evidence behind one genome's prediction.

        Distinguishing a detected resistance gene from a bare statistical association is
        a grading criterion — and the honest thing to show a clinician. A non-zero
        coefficient is not proof of biological causation, so a feature the model likes
        but which is not a curated determinant for this drug class is reported as an
        association, not a mechanism.
        """
        active = [name for name, coef in self.selected_features() if row.get(name, 0) == 1 and coef > 0]
        if not active:
            return EVIDENCE_NO_SIGNAL, []

        if self.drug is not None:
            class_tags = set(class_features_for_drug(self.drug))
            on_class = any(row.get(tag, 0) == 1 for tag in class_tags)
            known = [f for f in active if f in class_tags or f.startswith(("gene:", "point:"))]
            if known and on_class:
                return EVIDENCE_KNOWN_DETERMINANT, known[:5]

        return EVIDENCE_STATISTICAL, active[:5]

    def predict_one(self, row: pd.Series, species: str) -> Prediction:
        """Full decision path for one genome: gate -> model -> calibrate -> no-call."""
        self._check_fitted()

        if self.drug is not None:
            applicable, reason = self.drug.is_applicable(species)
            if not applicable:
                return Prediction(
                    drug=self.drug.name,
                    decision=LIKELY_TO_FAIL,
                    resistance_probability=0.99,
                    confidence=0.99,
                    evidence_type=EVIDENCE_INTRINSIC,
                    supporting_features=[],
                    gate_note=reason,
                )

        frame = row.to_frame().T[self.feature_names_].astype(int)
        probability = float(self.predict_proba(frame)[0])
        decision = str(apply_no_call(np.array([probability]), self.low, self.high)[0])
        evidence_type, features = self.explain(row)

        gate_note = ""

        if evidence_type == EVIDENCE_NO_SIGNAL:
            # Coherence rule: a "likely to fail" call must rest on a detected
            # determinant. Without one, a high probability reflects only the training
            # cohort's resistance prevalence — the model is reporting its prior, not
            # evidence from this genome. Asserting failure on that basis is the
            # false-confidence failure the brief warns against, so it is downgraded.
            if decision == LIKELY_TO_FAIL:
                decision = NO_CALL
                gate_note = (
                    "No resistance determinant was detected in this genome. The model's "
                    "elevated score reflects the resistance prevalence of the training "
                    "cohort rather than evidence from this isolate, so no call is made."
                )
            elif decision == NO_CALL:
                gate_note = (
                    "No resistance determinant was detected, and the model's score is "
                    "too close to the decision boundary to call either way."
                )
            else:
                gate_note = (
                    "No resistance determinant found. Absence of evidence is weaker than "
                    "evidence of susceptibility — confirm with standard lab testing."
                )
        elif self.drug is not None:
            _, reason = self.drug.is_applicable(species)
            gate_note = reason

        # Confidence is defined only for an actual call. Reporting a number for a
        # no-call invites reading it as certainty about the refusal; worse, any
        # closeness-to-boundary measure peaks exactly where the model knows least.
        if decision == LIKELY_TO_FAIL:
            confidence = probability
        elif decision == LIKELY_TO_WORK:
            confidence = 1.0 - probability
        else:
            confidence = None

        return Prediction(
            drug=self.drug.name if self.drug else self.drug_name,
            decision=decision,
            resistance_probability=probability,
            confidence=confidence,
            evidence_type=evidence_type,
            supporting_features=features,
            gate_note=gate_note,
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"DrugModel({self.drug_name}) is not fitted yet")


class GenomeFirewall:
    """The full panel: one DrugModel per antibiotic, trained and applied together."""

    def __init__(self, species: str, C: float = 0.1, low: float = 0.35, high: float = 0.65):
        self.species = species
        self.C = C
        self.low = low
        self.high = high
        self.models: dict[str, DrugModel] = {}
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_cal: pd.DataFrame | None = None,
        y_cal: pd.DataFrame | None = None,
        min_positive: int = 10,
    ) -> "GenomeFirewall":
        """Train one model per antibiotic column in `y_train`.

        Drugs with too few examples of either class are skipped rather than fitted on
        near-degenerate labels — a model trained on 3 resistant isolates would produce
        a confidence number with nothing behind it.
        """
        self.feature_names_ = list(X_train.columns)

        for drug in y_train.columns:
            labels = y_train[drug]
            observed = labels.notna()
            n_pos = int((labels[observed] == 1).sum())
            n_neg = int((labels[observed] == 0).sum())

            if n_pos < min_positive or n_neg < min_positive:
                print(f"  skip {drug}: {n_pos} resistant / {n_neg} susceptible (need >= {min_positive} each)")
                continue

            model = DrugModel(drug, C=self.C, low=self.low, high=self.high)
            train_ids = labels[observed].index

            cal_X = cal_y = None
            if X_cal is not None and y_cal is not None and drug in y_cal.columns:
                cal_mask = y_cal[drug].notna()
                if cal_mask.sum() >= 20:
                    cal_ids = y_cal[drug][cal_mask].index
                    cal_X = X_cal.loc[cal_ids]
                    cal_y = y_cal[drug][cal_mask].to_numpy(dtype=int)

            model.fit(X_train.loc[train_ids], labels[observed].to_numpy(dtype=int), cal_X, cal_y)
            self.models[drug] = model
            print(f"  fit  {drug}: {n_pos} resistant / {n_neg} susceptible, "
                  f"{len(model.selected_features())} determinants selected")

        return self

    def _check_fitted(self) -> None:
        if not self.models:
            raise RuntimeError(
                "GenomeFirewall is not fitted — no drug models were trained. Silently "
                "returning an empty report would read as 'no resistance found'."
            )

    def predict_genome(self, features: dict[str, int] | pd.Series) -> list[Prediction]:
        """Predict every trained drug for one genome's feature vector."""
        self._check_fitted()
        if isinstance(features, dict):
            row = pd.Series({name: features.get(name, 0) for name in self.feature_names_})
        else:
            row = features.reindex(self.feature_names_).fillna(0).astype(int)

        return [model.predict_one(row, self.species) for model in self.models.values()]

    def predict_cohort(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calibrated failure probabilities for a cohort: genomes x drugs."""
        self._check_fitted()
        return pd.DataFrame(
            {drug: model.predict_proba(X) for drug, model in self.models.items()},
            index=X.index,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved model panel ({len(self.models)} drugs) -> {path}")

    @staticmethod
    def load(path: str | Path) -> "GenomeFirewall":
        return joblib.load(Path(path))
