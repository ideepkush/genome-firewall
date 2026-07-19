"""Train the shipped panel on the real S. aureus cohort.

    python train_real.py

Inputs (data/real/):
    features.parquet   1,863 genomes x 137 AMRFinderPlus determinants
    labels.csv         long format, one row per genome-antibiotic pair, R/S
    groups.csv         the supplied homology grouping

Why the supplied grouping is not used as given
----------------------------------------------
groups.csv assigns 1,861 groups to 1,863 genomes — 99.9% singletons, largest group 3.
A "grouped" split over singletons is a random split, so it offers no protection against
near-identical isolates landing on both sides. S. aureus is highly clonal and that
grouping is not plausible.

We re-cluster on the determinant profile instead (543 groups, largest 117). It costs
0.171 balanced accuracy on tetracycline — 0.966 against 0.796 — and that gap is the
point: the higher number is the one a random split would have let us publish.

This is a substitute, not the real thing. The brief asks for sequence-homology
de-duplication, which needs assemblies this dataset does not ship.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.decision_report import (
    calibration_report,
    evaluate_panel,
    evidence_summary,
    generalization_report,
    plot_reliability,
    write_report,
)
from src.predictor import GenomeFirewall
from src.utils.clustering import cluster_from_feature_matrix, grouped_split, verify_no_leakage

DATA = Path("data/real")
OUT = Path("artifacts")
SPECIES = "staphylococcus aureus"
SEED = 42


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_parquet(DATA / "features.parquet")
    features.index = features.index.astype(str)
    # The supplied matrix uses GENE:/MUT:; the pipeline and drug knowledge base use
    # gene:/point:. Normalise here so determinant names match the curated lists.
    features.columns = [
        c.replace("GENE:", "gene:").replace("MUT:", "point:") for c in features.columns
    ]

    long = pd.read_csv(DATA / "labels.csv")
    labels = (
        long.assign(label=long.label.map({"R": 1.0, "S": 0.0}))
        .pivot_table(index="genome_id", columns="antibiotic", values="label", aggfunc="first")
    )
    labels.index = labels.index.astype(str)

    shared = features.index.intersection(labels.index)
    return features.loc[shared], labels.loc[shared]


def main() -> None:
    features, labels = load()
    print(f"Real cohort: {len(features):,} genomes x {features.shape[1]} determinants, "
          f"{labels.shape[1]} drugs")

    supplied = pd.read_csv(DATA / "groups.csv")
    print(f"Supplied grouping: {supplied.group_id.nunique():,} groups over "
          f"{len(supplied):,} genomes — {100 * (supplied.group_id.value_counts() == 1).mean():.1f}% "
          f"singletons, so not usable as a grouped split")

    groups = cluster_from_feature_matrix(features, threshold=0.95)
    print(f"Re-clustered on determinant profile: {groups.nunique()} groups, "
          f"largest {groups.value_counts().max()}")

    split = grouped_split(groups, test_frac=0.25, calibration_frac=0.15, seed=SEED)
    verify_no_leakage(split, groups)
    print(f"Leak-free split: {split.summary()}\n")

    panel = GenomeFirewall(species=SPECIES, C=0.1, low=0.35, high=0.65)
    panel.fit(features.loc[split.train], labels.loc[split.train],
              features.loc[split.calibration], labels.loc[split.calibration])

    X_test, y_test = features.loc[split.test], labels.loc[split.test]
    metrics = evaluate_panel(panel, X_test, y_test)
    print("\nHeld-out performance:")
    print(metrics[["n_test", "balanced_accuracy", "recall_resistant",
                   "recall_susceptible", "pr_auc", "brier", "no_call_rate"]].round(3).to_string())

    calibration, curves = calibration_report(panel, X_test, y_test)
    generalization = generalization_report(panel, X_test, y_test, groups)

    OUT.mkdir(parents=True, exist_ok=True)
    panel.save(OUT / "genome_firewall.joblib")
    write_report(metrics, calibration, generalization, evidence_summary(panel), OUT)
    plot_reliability(curves, OUT / "reliability.png")

    print("\nTetracycline recall on resistant isolates is the number to watch: roughly "
          "four in ten resistant isolates go unflagged, which is the unsafe direction.")


if __name__ == "__main__":
    main()
