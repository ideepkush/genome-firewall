"""Generates genome_firewall_colab.ipynb — a self-contained analysis notebook for Colab.

    python notebooks/build_colab_notebook.py

Design notes:
  * Colab has no conda, so AMRFinderPlus cannot run there. The notebook works from
    determinant feature matrices, which is the right input for model analysis anyway.
  * The notebook retrains rather than unpickling a saved panel. A joblib pickle carries
    the exact scikit-learn version it was written with, and Colab's version will not
    match; retraining on the same data takes seconds and sidesteps that entirely.
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda t: nbf.v4.new_markdown_cell(t.strip())
code = lambda s: nbf.v4.new_code_cell(s.strip())

nb.cells = [
    md("""
# 🧬 Genome Firewall — analysis notebook

Apply and analyse the antibiotic-resistance model in Google Colab.

**Runs here:** loading data, training, applying the model to genomes, metrics,
calibration, generalization, plots.

**Does not run here:** AMRFinderPlus annotation of raw FASTA files — it needs conda,
which Colab does not provide. This notebook works from *determinant feature matrices*
(the output of Module 01), which is the correct input for model analysis anyway. Run
annotation locally and bring the feature matrix here.

Run the cells in order. Setup is cells 1–3.
"""),
    md("## 1. Install dependencies"),
    code("""
!pip install -q scikit-learn pandas numpy matplotlib joblib
print("dependencies ready")
"""),
    md("""
## 2. Load the project code

Two ways to get `src/` into the runtime. Use whichever applies.

**A — clone from GitHub** (once your repo is pushed): set `REPO_URL` and run.

**B — upload a zip**: zip the project locally and upload when prompted.

```bash
cd ~/Desktop && zip -r genome-firewall.zip genome-firewall \\
  -x "*/.venv/*" "*/__pycache__/*" "*/.git/*"
```
"""),
    code("""
import os, sys, zipfile
from pathlib import Path

REPO_URL = ""   # e.g. "https://github.com/<you>/genome-firewall.git"
PROJECT = None

if REPO_URL:
    !git clone -q $REPO_URL /content/genome-firewall
    PROJECT = Path("/content/genome-firewall")
else:
    from google.colab import files
    print("Upload genome-firewall.zip …")
    uploaded = files.upload()
    name = next(iter(uploaded))
    with zipfile.ZipFile(name) as z:
        z.extractall("/content")
    # The zip may or may not contain a top-level folder.
    candidates = [p for p in Path("/content").rglob("src/predictor.py")]
    if not candidates:
        raise FileNotFoundError("Could not find src/predictor.py in the upload")
    PROJECT = candidates[0].parent.parent

sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)
print("project root:", PROJECT)
print("contents:", sorted(p.name for p in PROJECT.iterdir()))
"""),
    md("## 3. Imports"),
    code("""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.decision_report import (
    calibration_report, evaluate_panel, evidence_summary, generalization_report,
)
from src.predictor import GenomeFirewall
from src.utils.clustering import grouped_split, verify_no_leakage

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
print("imports OK")
"""),
    md("""
## 4. Load your data

`USE_SYNTHETIC = True` generates a cohort with the same hazards as real data (clonal
lineage structure, class imbalance, missing labels) so you can exercise everything
immediately.

Switch to `False` and upload three files to analyse a real cohort:

| File | Shape | Meaning |
|---|---|---|
| `features` | genomes × determinants | 1/0 — is this determinant present |
| `labels` | genomes × drugs | 1 = resistant, 0 = susceptible, blank = not tested |
| `clusters` | genomes → cluster_id | genetic lineage grouping |

All three must share the same genome IDs as their index.
"""),
    code("""
USE_SYNTHETIC = True
SPECIES = "escherichia coli"
SEED = 42

if USE_SYNTHETIC:
    from src.synthetic_data import generate_cohort
    features, labels, clusters = generate_cohort(n_genomes=1200, n_clusters=40, seed=SEED)
else:
    from google.colab import files
    print("Upload features.csv, labels.csv, clusters.csv …")
    files.upload()
    features = pd.read_csv("features.csv", index_col=0)
    labels   = pd.read_csv("labels.csv", index_col=0)
    clusters = pd.read_csv("clusters.csv", index_col=0).iloc[:, 0]
    shared = features.index.intersection(labels.index)
    features, labels, clusters = features.loc[shared], labels.loc[shared], clusters.loc[shared]

print(f"{features.shape[0]} genomes x {features.shape[1]} determinants")
print(f"{labels.shape[1]} drugs, {clusters.nunique()} clusters")
features.head()
"""),
    md("""
## 5. Split by genetic lineage

Whole clusters go to one split. Near-identical isolates in both train and test would
measure memorization rather than resistance biology, so `verify_no_leakage` raises if any
cluster spans two sets — before training, not after.

The calibration set is separate from test as well: fitting the calibrator on test data
would make the Brier score and reliability curve optimistic.
"""),
    code("""
split = grouped_split(clusters, test_frac=0.25, calibration_frac=0.15, seed=SEED)
verify_no_leakage(split, clusters)
print("no leakage —", split.summary())

X_train, y_train = features.loc[split.train], labels.loc[split.train]
X_cal,   y_cal   = features.loc[split.calibration], labels.loc[split.calibration]
X_test,  y_test  = features.loc[split.test], labels.loc[split.test]
"""),
    md("""
## 6. Train

One L1-penalized logistic regression per antibiotic. Knobs worth turning:

- `C` — lower means stronger regularization and fewer selected determinants
- `low` / `high` — the no-call band; widen it to make the system more cautious
"""),
    code("""
panel = GenomeFirewall(species=SPECIES, C=0.1, low=0.35, high=0.65)
panel.fit(X_train, y_train, X_cal, y_cal)
print(f"\\ntrained {len(panel.models)} drug models")
"""),
    md("""
---

# Applying the model

Everything below is analysis. Start here on re-runs.
"""),
    md("""
## 7. Apply to a single genome

Pass a dict of determinants. Each result carries its decision, calibrated confidence, the
evidence category, and the specific determinant behind it.
"""),
    code("""
def report(title, determinants):
    print(title)
    print("-" * len(title))
    for p in panel.predict_genome(determinants):
        figure = f"{p.confidence:.0%}" if p.is_called else f"p={p.resistance_probability:.0%}"
        print(f"  {p.drug:32} {p.decision:16} {figure:>6}  {p.evidence_type}")
        if p.supporting_features:
            print(f"  {'':32} evidence: {', '.join(p.supporting_features)}")
        if p.gate_note:
            print(f"  {'':32} note: {p.gate_note[:95]}")
    print()

report("ESBL-positive isolate", {"gene:blaCTX-M-15": 1, "class:BETA-LACTAM": 1})
report("No determinants detected", {})
"""),
    md("""
### Try your own

Available determinants are the columns of `features`. Edit and re-run.
"""),
    code("""
print("available determinants:")
print(sorted(c for c in features.columns if c.startswith(("gene:", "point:")))[:30])

my_genome = {
    "gene:blaKPC-2": 1,
    "class:BETA-LACTAM": 1,
    "point:gyrA_S83L": 1,
    "class:QUINOLONE": 1,
}
report("My isolate", my_genome)
"""),
    md("""
## 8. Apply across a cohort

`predict_cohort` returns calibrated failure probabilities for every genome × drug.
"""),
    code("""
probabilities = panel.predict_cohort(X_test)
print("probabilities:", probabilities.shape)
probabilities.head()
"""),
    code("""
fig, ax = plt.subplots(figsize=(9, 3.6))
for drug in probabilities.columns:
    ax.hist(probabilities[drug], bins=25, alpha=0.45, label=drug)
ax.axvspan(panel.low, panel.high, color="grey", alpha=0.18)
ax.text((panel.low + panel.high) / 2, ax.get_ylim()[1] * 0.9, "no-call band",
        ha="center", fontsize=9, color="#444")
ax.set_xlabel("calibrated probability the drug fails")
ax.set_ylabel("genomes")
ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
"""),
    md("""
## 9. Metrics on held-out lineages

`no_call_rate` is how often the system declines; `accuracy_on_called` is how it does on
the rest. A higher no-call rate with higher accuracy on the calls made is a system that
knows its limits, not a weaker one.
"""),
    code("""
metrics = evaluate_panel(panel, X_test, y_test)
metrics[[
    "n_test", "balanced_accuracy", "recall_resistant", "recall_susceptible",
    "f1", "auroc", "pr_auc", "brier", "no_call_rate", "accuracy_on_called",
]].round(3)
"""),
    md("""
## 10. Is the confidence real?

A reliability curve bins predictions by stated confidence and plots the observed rate in
each bin. Points on the diagonal mean the number is trustworthy. Expected calibration
error summarizes the gap — below ~0.05 is generally well calibrated.
"""),
    code("""
calibration, curves = calibration_report(panel, X_test, y_test)
display(calibration.round(3))

n = len(curves); cols = min(3, n); rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.3 * rows), squeeze=False)
for ax, (drug, curve) in zip(axes.ravel(), curves.items()):
    ok = ~np.isnan(curve["observed"])
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax.plot(curve["predicted"][ok], curve["observed"][ok], "o-", c="#2a78d6")
    ax.set_title(drug, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("observed")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for ax in axes.ravel()[n:]:
    ax.axis("off")
plt.tight_layout(); plt.show()
"""),
    md("""
## 11. Generalization — the table that matters most

An aggregate score hides lineage-specific collapse. Read the **min** column: that is what
happens when the system meets a lineage unlike anything it trained on, which is the
situation it actually faces in a hospital.
"""),
    code("""
generalization = generalization_report(panel, X_test, y_test, clusters, min_group_size=15)
spread = generalization.groupby("drug")["balanced_accuracy"].agg(["min", "mean", "max", "count"])
spread["overall"] = metrics.balanced_accuracy
display(spread.round(3))

fig, ax = plt.subplots(figsize=(8, 3.4))
for i, (drug, row) in enumerate(spread.iterrows()):
    ax.plot([row["min"], row["max"]], [i, i], c="#c3c2b7", lw=3, zorder=1)
    ax.scatter(row["overall"], i, c="#e34948", zorder=3, label="overall" if i == 0 else "")
    ax.scatter(row["min"], i, c="#2a78d6", zorder=2, label="worst lineage" if i == 0 else "")
ax.axvline(0.5, ls="--", c="grey", lw=1)
ax.set_yticks(range(len(spread))); ax.set_yticklabels(spread.index)
ax.set_xlabel("balanced accuracy"); ax.set_xlim(0.3, 1.02)
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout(); plt.show()
"""),
    md("""
## 12. What the models learned

L1 zeroes most coefficients, leaving a short list of named determinants per drug.
Positive weight pushes toward resistance.

A non-zero coefficient is **not** proof of a mechanism — a determinant riding the same
plasmid as the real cause also gets weight. That is why the report separates "known
determinant" from "statistical association only".
"""),
    code("""
evidence_summary(panel).head(25)
"""),
    md("""
## 13. Tuning sandbox

Sweep the regularization strength and watch the accuracy/interpretability trade-off:
stronger regularization (lower `C`) selects fewer determinants.
"""),
    code("""
rows = []
for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
    p = GenomeFirewall(species=SPECIES, C=C)
    p.fit(X_train, y_train, X_cal, y_cal)
    m = evaluate_panel(p, X_test, y_test)
    rows.append({
        "C": C,
        "mean_balanced_accuracy": m.balanced_accuracy.mean(),
        "mean_pr_auc": m.pr_auc.mean(),
        "mean_brier": m.brier.mean(),
        "mean_determinants": np.mean([len(mm.selected_features()) for mm in p.models.values()]),
    })
pd.DataFrame(rows).set_index("C").round(3)
"""),
    md("""
### Your own analysis

Objects available: `panel`, `features`, `labels`, `clusters`, `split`, `probabilities`,
`metrics`, `generalization`.

Useful calls:

```python
panel.predict_genome({...})              # one genome -> list[Prediction]
panel.predict_cohort(X)                  # many genomes -> DataFrame of probabilities
panel.models["ampicillin"].selected_features()   # learned determinants + coefficients
```
"""),
    code("""
# scratch


"""),
    md("""
---

## Saving results

Files written in Colab vanish when the runtime disconnects — download anything you want
to keep.
"""),
    code("""
from google.colab import files

metrics.to_csv("metrics_per_drug.csv")
generalization.to_csv("generalization_by_cluster.csv")
evidence_summary(panel).to_csv("selected_determinants.csv")

for f in ["metrics_per_drug.csv", "generalization_by_cluster.csv", "selected_determinants.csv"]:
    files.download(f)
"""),
]

out = Path(__file__).parent / "genome_firewall_colab.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(nb.cells)} cells)")
