"""Module 03 — Decision Report: evaluation, calibration checks, generalization.

Everything here answers one of the challenge's grading questions:
  * does the model work?              -> per-drug metric table
  * does its confidence mean anything? -> reliability curve, ECE, Brier
  * does it generalize?               -> metrics broken down by genetic cluster
  * can it admit ignorance?           -> no-call rate and accuracy on called subset
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .predictor import GenomeFirewall
from .utils.calibration import (
    DrugMetrics,
    evaluate_drug,
    expected_calibration_error,
    reliability_curve,
)


def evaluate_panel(
    panel: GenomeFirewall,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> pd.DataFrame:
    """Per-drug metrics on the held-out test set."""
    rows: list[DrugMetrics] = []
    for drug, model in panel.models.items():
        if drug not in y_test.columns:
            continue
        observed = y_test[drug].notna()
        if observed.sum() == 0:
            continue

        ids = y_test[drug][observed].index
        probabilities = model.predict_proba(X_test.loc[ids])
        rows.append(
            evaluate_drug(
                drug=drug,
                y_true=y_test.loc[ids, drug].to_numpy(dtype=int),
                probabilities=probabilities,
                low=model.low,
                high=model.high,
            )
        )

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame([m.to_dict() for m in rows]).set_index("drug")
    return table.sort_values("balanced_accuracy", ascending=False)


def calibration_report(
    panel: GenomeFirewall,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Expected calibration error per drug, plus the raw reliability curves for plotting."""
    summary, curves = [], {}
    for drug, model in panel.models.items():
        if drug not in y_test.columns:
            continue
        observed = y_test[drug].notna()
        if observed.sum() < n_bins:
            continue

        ids = y_test[drug][observed].index
        y_true = y_test.loc[ids, drug].to_numpy(dtype=int)
        probabilities = model.predict_proba(X_test.loc[ids])

        summary.append(
            {
                "drug": drug,
                "calibration_method": model.calibrator.fitted_method_,
                "expected_calibration_error": expected_calibration_error(y_true, probabilities, n_bins),
                "mean_predicted": float(probabilities.mean()),
                "observed_resistance_rate": float(y_true.mean()),
            }
        )
        curves[drug] = reliability_curve(y_true, probabilities, n_bins)

    return pd.DataFrame(summary).set_index("drug") if summary else pd.DataFrame(), curves


def generalization_report(
    panel: GenomeFirewall,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    clusters: pd.Series,
    min_group_size: int = 15,
) -> pd.DataFrame:
    """Metrics broken down by genetic cluster.

    A model that scores well overall but collapses on one lineage is memorizing that
    lineage's background, not learning resistance. Small clusters are skipped because a
    balanced accuracy computed on 4 isolates is noise.
    """
    records = []
    for drug, model in panel.models.items():
        if drug not in y_test.columns:
            continue

        observed = y_test[drug].notna()
        ids = y_test[drug][observed].index
        if len(ids) == 0:
            continue

        y_true = y_test.loc[ids, drug]
        probabilities = pd.Series(model.predict_proba(X_test.loc[ids]), index=ids)
        group = clusters.reindex(ids)

        for cluster_id, members in group.groupby(group):
            member_ids = members.index
            if len(member_ids) < min_group_size:
                continue
            if y_true.loc[member_ids].nunique() < 2:
                continue

            metrics = evaluate_drug(
                drug=drug,
                y_true=y_true.loc[member_ids].to_numpy(dtype=int),
                probabilities=probabilities.loc[member_ids].to_numpy(),
                low=model.low,
                high=model.high,
            )
            record = metrics.to_dict()
            record["cluster_id"] = int(cluster_id)
            records.append(record)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index(["drug", "cluster_id"]).sort_index()


def evidence_summary(panel: GenomeFirewall) -> pd.DataFrame:
    """What each drug model actually learned — its selected determinants.

    Shown in the demo to make the point that these are interpretable coefficients over
    named genes, not an opaque embedding. Positive weight pushes toward resistance.
    """
    rows = []
    for drug, model in panel.models.items():
        for rank, (feature, coefficient) in enumerate(model.selected_features()[:10], 1):
            rows.append(
                {
                    "drug": drug,
                    "rank": rank,
                    "determinant": feature,
                    "coefficient": round(coefficient, 4),
                    "direction": "resistance" if coefficient > 0 else "susceptibility",
                }
            )
    return pd.DataFrame(rows).set_index(["drug", "rank"]) if rows else pd.DataFrame()


def plot_reliability(curves: dict[str, dict], out_path: str | Path) -> None:
    """Reliability plot: predicted confidence vs observed resistance rate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not curves:
        return

    n = len(curves)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)

    for ax, (drug, curve) in zip(axes.ravel(), curves.items()):
        valid = ~np.isnan(curve["observed"])
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect")
        ax.plot(curve["predicted"][valid], curve["observed"][valid], "o-", color="#2a78d6", label="model")
        ax.set_title(drug, fontsize=10)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Reliability plot -> {out_path}")


def write_report(
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    generalization: pd.DataFrame,
    evidence: pd.DataFrame,
    out_dir: str | Path,
) -> None:
    """Persist every table the submission needs to show."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in (
        ("metrics_per_drug.csv", metrics),
        ("calibration_per_drug.csv", calibration),
        ("generalization_by_cluster.csv", generalization),
        ("selected_determinants.csv", evidence),
    ):
        if not frame.empty:
            frame.to_csv(out_dir / name)

    print(f"Reports -> {out_dir}")
