"""End-to-end training pipeline for Genome Firewall.

    python train.py --synthetic                       # verify the pipeline runs
    python train.py --features data/processed/features.parquet \
                    --labels data/raw/labels.csv \
                    --species "escherichia coli"      # real challenge dataset

The split is always at the genetic-cluster level. A random row split on this kind of
data reports a number that does not survive contact with a new isolate.
"""

from __future__ import annotations

import argparse
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


def load_inputs(args):
    """Return (features, labels, clusters, split_or_None).

    `split` is non-None only when the organizer pinned one. Their split is preferred
    over anything we derive: it is what makes the reported score comparable between
    teams rather than a product of each team's own choice of difficulty.
    """
    if args.synthetic:
        from src.synthetic_data import generate_cohort

        print("Generating synthetic cohort (pipeline verification only)")
        features, labels, clusters = generate_cohort(n_genomes=args.n_genomes, seed=args.seed)
        return features, labels, clusters, None

    from src.data_loader import align, load_features, load_groups, load_labels, load_split

    features = load_features(args.features, min_prevalence=args.min_prevalence)
    print()
    labels = load_labels(args.labels, lab_measured_only=not args.allow_predicted_labels)
    print()
    features, labels = align(features, labels)

    if args.clusters:
        clusters = load_groups(args.clusters, index=features.index)
        missing = clusters.isna()
        if missing.any():
            print(f"  {int(missing.sum()):,} genomes have no group; dropping them rather "
                  f"than assigning each its own, which would silently defeat the "
                  f"grouped split")
            keep = clusters[~missing].index
            features, labels, clusters = features.loc[keep], labels.loc[keep], clusters.loc[keep]
    else:
        print("No genetic-group file given — falling back to clustering on the "
              "resistance profile. This is weaker than the organizer's grouping; use "
              "--clusters when it ships.")
        clusters = cluster_from_feature_matrix(features)

    split = load_split(args.split, index=features.index) if args.split else None
    return features, labels, clusters, split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Genome Firewall panel")
    parser.add_argument("--synthetic", action="store_true", help="use generated data")
    parser.add_argument("--n-genomes", type=int, default=1200)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--clusters", type=Path, help="genome_id -> genetic group csv")
    parser.add_argument("--split", type=Path,
                        help="organizer's genome_id -> train/calibration/test csv; "
                             "used verbatim instead of deriving our own")
    parser.add_argument("--min-prevalence", type=int, default=3,
                        help="drop determinants seen in fewer than this many genomes")
    parser.add_argument("--allow-predicted-labels", action="store_true",
                        help="keep computationally-predicted phenotypes. Off by default: "
                             "training on predicted labels fits a model to another "
                             "model's output and every metric stays healthy while "
                             "meaning nothing.")
    parser.add_argument("--species", default="escherichia coli")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--C", type=float, default=0.1, help="inverse L1 regularization strength")
    parser.add_argument("--low", type=float, default=0.35, help="below this -> likely to work")
    parser.add_argument("--high", type=float, default=0.65, help="above this -> likely to fail")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.synthetic and (args.features is None or args.labels is None):
        parser.error("--features and --labels are required unless --synthetic is set")

    features, labels, clusters, given_split = load_inputs(args)
    print(f"\nCohort: {features.shape[0]} genomes x {features.shape[1]} features, "
          f"{labels.shape[1]} drugs, {clusters.nunique()} genetic clusters")

    if given_split is not None:
        split = given_split
        print("Using the organizer's split as given.")
    else:
        split = grouped_split(clusters, seed=args.seed)
        print(f"Cluster-disjoint split: {split.summary()}")
    verify_no_leakage(split, clusters)

    if not split.test:
        raise SystemExit(
            "The split has no test rows, so there is nothing to evaluate on. If the "
            "test set is hidden, train with --split omitted to derive a local held-out "
            "split for development, then submit predictions for the hidden set."
        )

    X_train, y_train = features.loc[split.train], labels.loc[split.train]
    X_cal, y_cal = features.loc[split.calibration], labels.loc[split.calibration]
    X_test, y_test = features.loc[split.test], labels.loc[split.test]

    print("\nTraining one model per antibiotic:")
    panel = GenomeFirewall(species=args.species, C=args.C, low=args.low, high=args.high)
    panel.fit(X_train, y_train, X_cal, y_cal)

    if not panel.models:
        raise SystemExit("No drug had enough labelled examples of both classes to train on.")

    print("\nEvaluating on held-out clusters:")
    metrics = evaluate_panel(panel, X_test, y_test)
    print(metrics[["balanced_accuracy", "recall_resistant", "recall_susceptible",
                   "pr_auc", "brier", "no_call_rate"]].round(3).to_string())

    calibration, curves = calibration_report(panel, X_test, y_test)
    if not calibration.empty:
        print("\nCalibration:")
        print(calibration.round(3).to_string())

    generalization = generalization_report(panel, X_test, y_test, clusters)
    if not generalization.empty:
        spread = generalization.groupby("drug")["balanced_accuracy"].agg(["min", "max", "count"])
        print("\nPer-cluster balanced accuracy (generalization spread):")
        print(spread.round(3).to_string())

    evidence = evidence_summary(panel)

    args.out.mkdir(parents=True, exist_ok=True)
    panel.save(args.out / "genome_firewall.joblib")
    write_report(metrics, calibration, generalization, evidence, args.out)
    plot_reliability(curves, args.out / "reliability.png")

    if args.synthetic:
        print("\nNOTE: these numbers come from synthetic data. They verify the pipeline "
              "runs correctly and prove nothing about real-world performance.")


if __name__ == "__main__":
    main()
