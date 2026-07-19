"""Confidence calibration and the evaluation metrics the challenge grades on.

A raw classifier score is not a probability. If the app is going to tell a clinician
"85% confident", then among all the calls it labels 85%, about 85% must actually be
right. That property is what these functions fit and then verify.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)

NO_CALL = "no-call"
LIKELY_TO_FAIL = "likely to fail"
LIKELY_TO_WORK = "likely to work"


class ProbabilityCalibrator:
    """Maps raw model scores to calibrated probabilities.

    Isotonic regression is the default: it is non-parametric and handles the
    S-shaped miscalibration typical of regularized logistic regression. It needs a few
    hundred calibration points, so we fall back to Platt scaling (a 1-D logistic fit)
    on small calibration sets, where isotonic would just overfit the noise.
    """

    def __init__(self, method: str = "auto", min_isotonic_samples: int = 200):
        self.method = method
        self.min_isotonic_samples = min_isotonic_samples
        self.fitted_method_: str | None = None
        self._model = None

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        scores = np.asarray(scores, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=int).ravel()

        method = self.method
        if method == "auto":
            enough_data = len(scores) >= self.min_isotonic_samples
            method = "isotonic" if enough_data else "platt"

        if len(np.unique(y_true)) < 2:
            # Only one class in the calibration set — no signal to calibrate against.
            self.fitted_method_ = "identity"
            self._model = None
            return self

        if method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._model.fit(scores, y_true)
        else:
            self._model = LogisticRegression(C=1e6, solver="lbfgs")
            self._model.fit(scores.reshape(-1, 1), y_true)

        self.fitted_method_ = method
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float).ravel()
        if self._model is None:
            return np.clip(scores, 0.0, 1.0)
        if self.fitted_method_ == "isotonic":
            return np.clip(self._model.predict(scores), 0.0, 1.0)
        return self._model.predict_proba(scores.reshape(-1, 1))[:, 1]


def apply_no_call(
    probabilities: np.ndarray, low: float = 0.35, high: float = 0.65
) -> np.ndarray:
    """Turn calibrated probabilities into three-way decisions.

    Probabilities in the uncertain band become `no-call`. Refusing to answer is the
    honest output when evidence is weak or conflicting — a confident wrong call sends
    a care team toward the wrong drug.
    """
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    decisions = np.full(probabilities.shape, NO_CALL, dtype=object)
    decisions[probabilities >= high] = LIKELY_TO_FAIL
    decisions[probabilities <= low] = LIKELY_TO_WORK
    return decisions


@dataclass
class DrugMetrics:
    """Everything the challenge asks to be reported, for one antibiotic."""

    drug: str
    n_test: int
    n_resistant: int
    n_susceptible: int
    balanced_accuracy: float
    recall_resistant: float
    recall_susceptible: float
    f1: float
    auroc: float
    pr_auc: float
    brier: float
    no_call_rate: float
    accuracy_on_called: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_drug(
    drug: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    low: float = 0.35,
    high: float = 0.65,
) -> DrugMetrics:
    """Score one antibiotic. Label convention: 1 = resistant (drug likely to fail)."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    hard_pred = (probabilities >= 0.5).astype(int)

    both_classes = len(np.unique(y_true)) == 2
    decisions = apply_no_call(probabilities, low, high)
    called = decisions != NO_CALL
    called_correct = (
        ((decisions[called] == LIKELY_TO_FAIL).astype(int) == y_true[called]).mean()
        if called.any()
        else float("nan")
    )

    return DrugMetrics(
        drug=drug,
        n_test=int(len(y_true)),
        n_resistant=int((y_true == 1).sum()),
        n_susceptible=int((y_true == 0).sum()),
        balanced_accuracy=float(balanced_accuracy_score(y_true, hard_pred)),
        recall_resistant=float(recall_score(y_true, hard_pred, pos_label=1, zero_division=0)),
        recall_susceptible=float(recall_score(y_true, hard_pred, pos_label=0, zero_division=0)),
        f1=float(f1_score(y_true, hard_pred, zero_division=0)),
        auroc=float(roc_auc_score(y_true, probabilities)) if both_classes else float("nan"),
        pr_auc=float(average_precision_score(y_true, probabilities)) if both_classes else float("nan"),
        brier=float(brier_score_loss(y_true, probabilities)),
        no_call_rate=float((~called).mean()),
        accuracy_on_called=float(called_correct),
    )


def reliability_curve(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray]:
    """Bin predictions by confidence and measure the observed hit rate in each bin.

    Perfect calibration means `observed` tracks `predicted` along the diagonal. The
    gap between them is what a reliability plot draws.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)

    predicted, observed, counts = [], [], []
    for b in range(n_bins):
        mask = bin_index == b
        counts.append(int(mask.sum()))
        if mask.any():
            predicted.append(float(probabilities[mask].mean()))
            observed.append(float(y_true[mask].mean()))
        else:
            predicted.append(np.nan)
            observed.append(np.nan)

    return {
        "bin_center": (edges[:-1] + edges[1:]) / 2,
        "predicted": np.array(predicted),
        "observed": np.array(observed),
        "count": np.array(counts),
    }


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """Weighted mean gap between predicted confidence and observed accuracy.

    A single number summarizing the reliability plot. Lower is better; below ~0.05 is
    generally considered well calibrated.
    """
    curve = reliability_curve(y_true, probabilities, n_bins)
    counts = curve["count"]
    total = counts.sum()
    if total == 0:
        return float("nan")

    gaps = np.abs(curve["predicted"] - curve["observed"])
    valid = ~np.isnan(gaps)
    return float((gaps[valid] * counts[valid]).sum() / total)
