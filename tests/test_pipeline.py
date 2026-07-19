"""Tests for the properties that make the reported metrics trustworthy.

These guard the claims the submission makes: no cluster leakage across splits, honest
no-call behaviour, a working intrinsic-resistance gate, and evidence attribution that
does not overstate what the model knows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.drug_database import get_drug
from src.genome_reader import hits_to_features
from src.predictor import (
    EVIDENCE_KNOWN_DETERMINANT,
    EVIDENCE_INTRINSIC,
    EVIDENCE_NO_SIGNAL,
    GenomeFirewall,
)
from src.synthetic_data import generate_cohort
from src.utils.amrfinder import AMRHit
from src.utils.calibration import (
    LIKELY_TO_FAIL,
    LIKELY_TO_WORK,
    NO_CALL,
    ProbabilityCalibrator,
    apply_no_call,
    expected_calibration_error,
)
from src.utils.clustering import grouped_split, verify_no_leakage


@pytest.fixture(scope="module")
def cohort():
    return generate_cohort(n_genomes=600, n_clusters=25, seed=7)


@pytest.fixture(scope="module")
def trained_panel(cohort):
    features, labels, clusters = cohort
    split = grouped_split(clusters, seed=7)
    panel = GenomeFirewall(species="escherichia coli")
    panel.fit(
        features.loc[split.train],
        labels.loc[split.train],
        features.loc[split.calibration],
        labels.loc[split.calibration],
    )
    return panel, features, labels, split


class TestSplitting:
    def test_split_is_cluster_disjoint(self, cohort):
        _, _, clusters = cohort
        split = grouped_split(clusters, seed=7)
        verify_no_leakage(split, clusters)  # raises if a cluster spans two sets

    def test_split_covers_every_genome_exactly_once(self, cohort):
        _, _, clusters = cohort
        split = grouped_split(clusters, seed=7)
        assigned = split.train + split.calibration + split.test
        assert len(assigned) == len(clusters)
        assert set(assigned) == set(clusters.index)

    def test_leakage_check_catches_a_bad_split(self, cohort):
        _, _, clusters = cohort
        split = grouped_split(clusters, seed=7)
        split.train.append(split.test[0])  # force one genome into both sets
        with pytest.raises(AssertionError, match="leakage"):
            verify_no_leakage(split, clusters)


class TestNoCall:
    def test_uncertain_band_returns_no_call(self):
        decisions = apply_no_call(np.array([0.05, 0.5, 0.95]), low=0.35, high=0.65)
        assert list(decisions) == ["likely to work", NO_CALL, LIKELY_TO_FAIL]

    def test_band_boundaries_are_inclusive(self):
        decisions = apply_no_call(np.array([0.35, 0.65]), low=0.35, high=0.65)
        assert list(decisions) == ["likely to work", LIKELY_TO_FAIL]


class TestCalibration:
    def test_calibration_improves_a_skewed_score(self):
        rng = np.random.default_rng(0)
        y = rng.binomial(1, 0.3, size=800)
        # Overconfident scores: pushed toward the extremes relative to true rate.
        raw = np.clip(y * 0.55 + 0.22 + rng.normal(0, 0.12, size=800), 0, 1)

        calibrator = ProbabilityCalibrator(method="auto").fit(raw, y)
        calibrated = calibrator.transform(raw)

        assert expected_calibration_error(y, calibrated) <= expected_calibration_error(y, raw)

    def test_single_class_calibration_does_not_crash(self):
        calibrator = ProbabilityCalibrator().fit(np.array([0.2, 0.4, 0.6]), np.array([1, 1, 1]))
        assert calibrator.fitted_method_ == "identity"
        assert np.all((calibrator.transform(np.array([0.5])) >= 0) & (calibrator.transform(np.array([0.5])) <= 1))


class TestTargetGate:
    def test_vancomycin_blocked_in_gram_negative(self):
        applicable, reason = get_drug("vancomycin").is_applicable("escherichia coli")
        assert not applicable
        assert "gram-negative" in reason

    def test_vancomycin_allowed_in_gram_positive(self):
        applicable, _ = get_drug("vancomycin").is_applicable("staphylococcus aureus")
        assert applicable

    def test_intrinsic_resistance_is_flagged(self):
        applicable, reason = get_drug("ampicillin").is_applicable("klebsiella pneumoniae")
        assert not applicable
        assert "intrinsically resistant" in reason

    def test_unknown_species_does_not_block(self):
        applicable, reason = get_drug("colistin").is_applicable("some novel species")
        assert applicable
        assert "not in the property table" in reason

    def test_drug_lookup_tolerates_aliases(self):
        assert get_drug("TMP-SMX") is get_drug("trimethoprim-sulfamethoxazole")
        assert get_drug("nonexistent-drug") is None


class TestAMRFinderSchema:
    """The parser must survive AMRFinderPlus's column renames.

    v4 renamed most headers. Reading a v4 report with only v3 names matched nothing and
    produced an empty hit list — indistinguishable from a genome with no resistance.
    """

    V4_HEADER = [
        "Protein id", "Contig id", "Element symbol", "Element name",
        "Type", "Subtype", "Class", "Subclass",
        "% Coverage of reference", "% Identity to reference",
    ]
    V3_HEADER = [
        "Protein identifier", "Contig id", "Gene symbol", "Sequence name",
        "Element type", "Element subtype", "Class", "Subclass",
        "% Coverage of reference sequence", "% Identity to reference sequence",
    ]
    ROW = ["NA", "contig1", "blaCTX-M-15", "ESBL", "AMR", "AMR",
           "BETA-LACTAM", "CEPHALOSPORIN", "100.0", "100.0"]

    @pytest.mark.parametrize("header", [V4_HEADER, V3_HEADER])
    def test_both_schema_versions_parse(self, header):
        from src.utils.amrfinder import normalize_report, parse_report

        hits = parse_report(normalize_report(pd.DataFrame([self.ROW], columns=header)))
        assert len(hits) == 1
        assert hits[0].gene_symbol == "blaCTX-M-15"
        assert hits[0].drug_class == "BETA-LACTAM"

    def test_unrecognized_schema_raises_instead_of_returning_nothing(self):
        from src.utils.amrfinder import normalize_report

        bogus = pd.DataFrame([["x", "y"]], columns=["Something", "Else"])
        with pytest.raises(ValueError, match="missing required column"):
            normalize_report(bogus)


class TestEvidenceVocabulary:
    """Drug class tags must match AMRFinderPlus's vocabulary.

    AMRFinderPlus reports sul1 under SULFONAMIDE. A drug database that called the same
    thing FOLATE-PATHWAY-ANTAGONIST failed to match, and a genuine known determinant was
    reported as a mere statistical association — understating real evidence.
    """

    def test_combination_drug_covers_each_component_class(self):
        from src.drug_database import class_features_for_drug

        tags = set(class_features_for_drug(get_drug("trimethoprim-sulfamethoxazole")))
        assert "class:SULFONAMIDE" in tags
        assert "class:TRIMETHOPRIM" in tags

    def test_sulfonamide_gene_reads_as_a_known_mechanism(self, trained_panel):
        panel, _, _, _ = trained_panel
        drug = "trimethoprim-sulfamethoxazole"
        if drug not in panel.models:
            pytest.skip(f"{drug} not trained in this sample")

        predictions = panel.predict_genome({"gene:sul1": 1, "class:SULFONAMIDE": 1})
        pred = predictions[list(panel.models).index(drug)]
        assert pred.evidence_type == EVIDENCE_KNOWN_DETERMINANT, (
            f"sul1 should count as a known mechanism, got: {pred.evidence_type}"
        )


class TestFeatureEncoding:
    def test_hits_become_gene_point_and_class_features(self):
        hits = [
            AMRHit("blaTEM-1", "AMR", "AMR", "BETA-LACTAM", "BETA-LACTAM", 99.0, 100.0, "c1"),
            AMRHit("gyrA_S83L", "AMR", "POINT", "QUINOLONE", "QUINOLONE", 99.5, 100.0, "c1"),
        ]
        features = hits_to_features(hits)
        assert features["gene:blaTEM-1"] == 1
        assert features["point:gyrA_S83L"] == 1
        assert features["class:BETA-LACTAM"] == 1
        assert features["class:QUINOLONE"] == 1

    def test_unnamed_hits_are_dropped(self):
        assert hits_to_features([AMRHit("", "AMR", "AMR", "", "", 99.0, 100.0, "c1")]) == {}


class TestPredictions:
    def test_panel_trains_on_every_well_populated_drug(self, trained_panel):
        panel, _, labels, _ = trained_panel
        assert len(panel.models) > 0
        assert set(panel.models).issubset(set(labels.columns))

    def test_prediction_carries_confidence_and_evidence(self, trained_panel):
        panel, features, _, split = trained_panel
        predictions = panel.predict_genome(features.loc[split.test[0]])

        assert len(predictions) == len(panel.models)
        for pred in predictions:
            assert 0.0 <= pred.resistance_probability <= 1.0
            assert pred.decision in {LIKELY_TO_FAIL, "likely to work", NO_CALL}
            assert pred.evidence_type
            if pred.is_called:
                assert 0.0 <= pred.confidence <= 1.0

    def test_no_call_reports_no_confidence(self, trained_panel):
        """A no-call has no claim to be confident about.

        The earlier formula scored a no-call by closeness to the decision boundary,
        which peaks exactly where the model knows least — a genome the system refused
        to call was displayed as "94% confident". Confidence is now undefined unless a
        decision was actually made.
        """
        panel, features, _, split = trained_panel

        seen_no_call = False
        for genome_id in split.test[:40]:
            for pred in panel.predict_genome(features.loc[genome_id]):
                if pred.decision == NO_CALL:
                    seen_no_call = True
                    assert pred.confidence is None, (
                        f"{pred.drug} reported confidence {pred.confidence} on a no-call"
                    )
                else:
                    assert pred.confidence is not None

        assert seen_no_call, "no no-call encountered; test did not exercise the path"

    def test_called_confidence_agrees_with_the_decision(self, trained_panel):
        """Confidence must describe the decision made, not always resistance."""
        panel, features, _, split = trained_panel

        for genome_id in split.test[:40]:
            for pred in panel.predict_genome(features.loc[genome_id]):
                if pred.decision == LIKELY_TO_FAIL:
                    assert pred.confidence == pytest.approx(pred.resistance_probability)
                elif pred.decision == LIKELY_TO_WORK:
                    assert pred.confidence == pytest.approx(1.0 - pred.resistance_probability)

    def test_empty_genome_never_asserts_failure(self, trained_panel):
        """The core honesty property.

        With no determinant detected, a high model score reflects only the training
        cohort's resistance prevalence. Reporting that as "likely to fail" would be
        confident-but-unevidenced — the exact failure the challenge penalizes.
        """
        panel, _, _, _ = trained_panel

        for pred in panel.predict_genome({}):
            assert pred.supporting_features == []
            if pred.evidence_type == EVIDENCE_NO_SIGNAL:
                assert pred.decision != LIKELY_TO_FAIL, (
                    f"{pred.drug} asserted failure with no supporting determinant"
                )

    def test_unevidenced_failure_is_downgraded_with_an_explanation(self, trained_panel):
        panel, _, _, _ = trained_panel
        downgraded = [
            p for p in panel.predict_genome({})
            if p.decision == NO_CALL and p.evidence_type == EVIDENCE_NO_SIGNAL
        ]
        for pred in downgraded:
            assert "training" in pred.gate_note.lower() or "no resistance" in pred.gate_note.lower()

    def test_detected_determinant_still_drives_a_failure_call(self, trained_panel):
        """The downgrade must not suppress genuine evidence-backed resistance."""
        panel, _, _, _ = trained_panel
        predictions = panel.predict_genome({"gene:blaCTX-M-15": 1, "class:BETA-LACTAM": 1})

        beta_lactams = [p for p in predictions if p.drug.lower() in {"ampicillin", "ceftriaxone"}]
        assert any(p.decision == LIKELY_TO_FAIL for p in beta_lactams), (
            "an ESBL gene should still produce a failure call for beta-lactams"
        )
        for pred in beta_lactams:
            if pred.decision == LIKELY_TO_FAIL:
                assert pred.supporting_features, "a failure call must cite its evidence"

    def test_intrinsic_gate_overrides_the_model(self):
        panel = GenomeFirewall(species="klebsiella pneumoniae")
        features, labels, clusters = generate_cohort(n_genomes=300, n_clusters=12, seed=3)
        split = grouped_split(clusters, seed=3)
        panel.fit(features.loc[split.train], labels.loc[split.train])

        if "ampicillin" not in panel.models:
            pytest.skip("ampicillin not trained in this sample")

        pred = panel.predict_genome({})[list(panel.models).index("ampicillin")]
        assert pred.decision == LIKELY_TO_FAIL
        assert pred.evidence_type == EVIDENCE_INTRINSIC

    def test_cohort_probabilities_are_valid(self, trained_panel):
        panel, features, _, split = trained_panel
        probabilities = panel.predict_cohort(features.loc[split.test])
        assert probabilities.shape == (len(split.test), len(panel.models))
        assert probabilities.to_numpy().min() >= 0.0
        assert probabilities.to_numpy().max() <= 1.0

    def test_unfitted_model_refuses_to_predict(self):
        panel = GenomeFirewall(species="escherichia coli")
        with pytest.raises(RuntimeError, match="not fitted"):
            panel.predict_cohort(pd.DataFrame({"gene:x": [1]}))
