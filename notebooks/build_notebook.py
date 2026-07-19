"""Generates evaluation.ipynb.

The notebook is kept as a build script so it stays diffable in git — committed .ipynb
files carry execution counts and base64 image blobs that make review painful.

    python notebooks/build_notebook.py && jupyter lab notebooks/evaluation.ipynb
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda text: nbf.v4.new_markdown_cell(text.strip())
code = lambda src: nbf.v4.new_code_cell(src.strip())

nb.cells = [
    md("""
# Genome Firewall — evaluation

Walks the full pipeline on one cohort and reports what the challenge grades on:
evidence, calibrated confidence, honest generalization, and a working no-call.

Run `python train.py --synthetic` first, or point `USE_SYNTHETIC = False` at the
challenge dataset once it is available.

**These numbers come from synthetic data unless you switch the flag below.** They verify
the pipeline is correct and prove nothing about real-world performance.
"""),
    code("""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.decision_report import (
    calibration_report,
    evaluate_panel,
    evidence_summary,
    generalization_report,
)
from src.predictor import GenomeFirewall
from src.utils.calibration import reliability_curve
from src.utils.clustering import grouped_split, verify_no_leakage

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

USE_SYNTHETIC = True
SPECIES = "escherichia coli"
SEED = 42
"""),
    md("## 1. Load the cohort"),
    code("""
if USE_SYNTHETIC:
    from src.synthetic_data import generate_cohort
    features, labels, clusters = generate_cohort(n_genomes=1200, n_clusters=40, seed=SEED)
else:
    features = pd.read_parquet("../data/processed/features.parquet")
    labels = pd.read_csv("../data/raw/labels.csv", index_col=0)
    clusters = pd.read_csv("../data/splits/genetic_groups.csv", index_col=0).iloc[:, 0]
    shared = features.index.intersection(labels.index)
    features, labels, clusters = features.loc[shared], labels.loc[shared], clusters.loc[shared]

print(f"{features.shape[0]} genomes x {features.shape[1]} determinants")
print(f"{labels.shape[1]} antibiotics, {clusters.nunique()} genetic clusters")
features.head()
"""),
    md("""
## 2. The two hazards in this data

**Class imbalance.** Resistance is far more common than susceptibility in curated
collections, because isolates get sequenced when they cause trouble. Raw accuracy is
therefore misleading — a model predicting "resistant" every time would score well.

**Lineage structure.** Genomes cluster into near-identical clonal groups. This is what
makes a random row split dishonest.
"""),
    code("""
summary = pd.DataFrame({
    "n_tested": labels.notna().sum(),
    "n_resistant": (labels == 1).sum(),
    "n_susceptible": (labels == 0).sum(),
})
summary["resistant_pct"] = (100 * summary.n_resistant / summary.n_tested).round(1)
summary["missing_pct"] = (100 * (1 - summary.n_tested / len(labels))).round(1)
summary
"""),
    code("""
sizes = clusters.value_counts()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].bar(summary.index, summary.resistant_pct, color="#2a78d6")
axes[0].axhline(50, ls="--", c="grey", lw=1)
axes[0].set_ylabel("% resistant")
axes[0].set_title("Class balance per drug")
axes[0].tick_params(axis="x", rotation=45)
for label in axes[0].get_xticklabels():
    label.set_ha("right")

axes[1].hist(sizes.values, bins=20, color="#1baf7a")
axes[1].set_xlabel("genomes per cluster")
axes[1].set_ylabel("clusters")
axes[1].set_title(f"Lineage structure ({clusters.nunique()} clusters)")
plt.tight_layout()
plt.show()

print(f"largest cluster: {sizes.max()} genomes ({100*sizes.max()/len(clusters):.1f}% of cohort)")
"""),
    md("""
## 3. Splitting by cluster, not by row

Whole clusters go to one split. `verify_no_leakage` raises if any cluster spans two
sets, and runs before training rather than after — a leak discovered post-hoc has
already contaminated every number downstream.

The calibration set is separate from test as well. Fitting the calibrator on test data
would make the Brier score and reliability curve optimistic, because the calibrator
would have seen its own answers.
"""),
    code("""
split = grouped_split(clusters, test_frac=0.25, calibration_frac=0.15, seed=SEED)
verify_no_leakage(split, clusters)
print("no leakage:", split.summary())

X_train, y_train = features.loc[split.train], labels.loc[split.train]
X_cal, y_cal = features.loc[split.calibration], labels.loc[split.calibration]
X_test, y_test = features.loc[split.test], labels.loc[split.test]
"""),
    md("""
### What a random split would have claimed

Worth quantifying rather than asserting. The cell below trains the same model under a
random row split and compares. The gap is the portion of the score that comes from
recognizing lineages the model has already seen.
"""),
    code("""
rng = np.random.default_rng(SEED)
shuffled = rng.permutation(features.index)
cut = int(0.75 * len(shuffled))
rand_train, rand_test = list(shuffled[:cut]), list(shuffled[cut:])

random_panel = GenomeFirewall(species=SPECIES)
random_panel.fit(features.loc[rand_train], labels.loc[rand_train])
random_metrics = evaluate_panel(random_panel, features.loc[rand_test], labels.loc[rand_test])

print("random row split (optimistic):")
print(random_metrics[["balanced_accuracy", "pr_auc"]].round(3).to_string())
"""),
    md("## 4. Train the panel"),
    code("""
panel = GenomeFirewall(species=SPECIES, C=0.1, low=0.35, high=0.65)
panel.fit(X_train, y_train, X_cal, y_cal)
"""),
    md("""
## 5. Metrics on held-out clusters

`no_call_rate` is how often the system declines to answer, and `accuracy_on_called` is
how it does on the rest. A higher no-call rate paired with higher accuracy on the calls
made is a system that knows its own limits, not a weaker one.
"""),
    code("""
metrics = evaluate_panel(panel, X_test, y_test)
metrics[[
    "n_test", "balanced_accuracy", "recall_resistant", "recall_susceptible",
    "f1", "auroc", "pr_auc", "brier", "no_call_rate", "accuracy_on_called",
]].round(3)
"""),
    code("""
comparison = pd.DataFrame({
    "cluster_split": metrics.balanced_accuracy,
    "random_split": random_metrics.balanced_accuracy,
}).dropna()
comparison["inflation"] = (comparison.random_split - comparison.cluster_split).round(3)

ax = comparison[["cluster_split", "random_split"]].plot.barh(
    figsize=(8, 3.6), color=["#2a78d6", "#e34948"]
)
ax.set_xlabel("balanced accuracy")
ax.set_xlim(0, 1)
ax.set_title("Honest split vs random split")
plt.tight_layout()
plt.show()

print(f"mean inflation from splitting randomly: {comparison.inflation.mean():+.3f}")
"""),
    md("""
## 6. Is the confidence real?

A reliability curve bins predictions by stated confidence and plots the observed rate in
each bin. Points on the diagonal mean the number shown to a clinician is trustworthy.
Expected calibration error summarizes the gap; below ~0.05 is generally well calibrated.
"""),
    code("""
calibration, curves = calibration_report(panel, X_test, y_test)
calibration.round(3)
"""),
    code("""
n = len(curves)
cols = min(3, n)
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), squeeze=False)

for ax, (drug, curve) in zip(axes.ravel(), curves.items()):
    valid = ~np.isnan(curve["observed"])
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="perfect")
    ax.plot(curve["predicted"][valid], curve["observed"][valid], "o-", c="#2a78d6", label="model")
    ax.set_title(drug, fontsize=10)
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
    md("""
## 7. Generalization — the table that matters most

A headline average hides lineage-specific collapse. Below is balanced accuracy per drug
broken down by genetic cluster. Read the **min** column: that is what happens when the
system meets a lineage unlike its training data, which is the situation it will actually
face in a hospital.
"""),
    code("""
generalization = generalization_report(panel, X_test, y_test, clusters, min_group_size=15)
spread = generalization.groupby("drug")["balanced_accuracy"].agg(["min", "mean", "max", "count"])
spread["overall"] = metrics.balanced_accuracy
spread.round(3)
"""),
    code("""
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, (drug, row) in enumerate(spread.iterrows()):
    ax.plot([row["min"], row["max"]], [i, i], c="#c3c2b7", lw=3, zorder=1)
    ax.scatter([row["overall"]], [i], c="#e34948", zorder=3, label="overall" if i == 0 else "")
    ax.scatter([row["min"]], [i], c="#2a78d6", zorder=2, label="worst cluster" if i == 0 else "")

ax.axvline(0.5, ls="--", c="grey", lw=1)
ax.set_yticks(range(len(spread)))
ax.set_yticklabels(spread.index)
ax.set_xlabel("balanced accuracy")
ax.set_xlim(0.3, 1.02)
ax.set_title("Overall score vs worst-cluster score (0.5 = chance)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.show()
"""),
    md("""
## 8. What the models learned

L1 regularization zeroes most coefficients, so each drug ends up with a short list of
named determinants. Positive weight pushes toward resistance.

A non-zero coefficient is **not** proof of a biological mechanism — a determinant that
merely travels on the same plasmid as the real cause will also get weight. This is why
the app separates "known resistance determinant detected" from "statistical association
only" rather than presenting every coefficient as an explanation.
"""),
    code("""
evidence_summary(panel).head(25)
"""),
    md("""
## 9. Two reports, end to end

The behaviour that matters: evidence-backed resistance produces a call with the gene
cited; an absence of evidence does not produce a confident claim in either direction.
"""),
    code("""
def show(title, genome_features):
    print(title)
    print("-" * len(title))
    for pred in panel.predict_genome(genome_features):
        confidence = f"{pred.confidence:.0%}" if pred.is_called else f"p={pred.resistance_probability:.0%}"
        print(f"  {pred.drug:32} {pred.decision:16} {confidence:>6}  {pred.evidence_type}")
        if pred.supporting_features:
            print(f"  {'':32} evidence: {', '.join(pred.supporting_features)}")
    print()

show("ESBL-positive isolate (blaCTX-M-15)", {"gene:blaCTX-M-15": 1, "class:BETA-LACTAM": 1})
show("No determinants detected", {})
"""),
    md("""
## 10. Summary

- Split at the cluster level, verified leak-free before training.
- Confidence calibrated on a split disjoint from both train and test.
- Every failure call cites a detected determinant; without one the system returns
  `no-call` rather than reporting the training cohort's base rate as a finding.
- Generalization reported per lineage, including the worst case, not just the average.

**Limits.** One species, the antibiotics listed above, and a defensive scope: this
predicts resistance that already exists. Every result requires laboratory confirmation
before it informs treatment.
"""),
]

out = Path(__file__).parent / "evaluation.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(nb.cells)} cells)")
