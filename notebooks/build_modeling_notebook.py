"""Generates modeling.ipynb — build, train, save, and hand the model to the frontend.

    python notebooks/build_modeling_notebook.py

This is the notebook-first workflow: the modelling is written out in cells rather than
hidden behind an import, so it can be read and changed in place. The last section packs
what the cells built into the panel object the Streamlit app expects and saves it, so the
app picks it up with no changes.

The from-scratch code and src/predictor.py must agree. Section 9 asserts that they do —
if a cell here drifts from the module the app serves, the notebook fails loudly instead
of shipping a model that disagrees with the demo.
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda t: nbf.v4.new_markdown_cell(t.strip())
code = lambda s: nbf.v4.new_code_cell(s.strip())

nb.cells = [
    md("""
# 🧬 Genome Firewall — build the model

The full workflow in one notebook: load data → explore → train → calibrate → evaluate →
**save** → the Streamlit app loads it.

Every modelling step is written out below rather than imported, so you can change the
regularization, the calibrator, or the decision thresholds and immediately see what
moves.

**One rule this notebook enforces on itself.** Section 9 checks that the model built here
produces the same predictions as `src/predictor.py`, the module the app actually serves.
Cells and modules that drift silently are how a demo ends up disagreeing with the
notebook it came from, so the check is an assert, not a comment.
"""),
    md("## 1. Setup"),
    code("""
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, recall_score, roc_auc_score,
)

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 40)

SPECIES = "staphylococcus aureus"
SEED = 42
C = 0.1        # inverse L1 strength — lower selects fewer determinants
LOW, HIGH = 0.35, 0.65   # outside this band we call; inside we return no-call

print("ready")
"""),
    md("""
## 2. Load the cohort

`USE_SYNTHETIC = True` generates a cohort carrying the same hazards as the real data:
clonal lineage structure, class imbalance, missing labels.

For the challenge dataset set it to `False` and point the paths at your files. The loader
handles long-format labels (one row per genome-antibiotic pair), maps S/I/R text, and
drops computationally-predicted rows — see `src/data_loader.py`.
"""),
    code("""
USE_SYNTHETIC = True

if USE_SYNTHETIC:
    from src.synthetic_data import generate_cohort
    features, labels, clusters = generate_cohort(n_genomes=1200, n_clusters=40, seed=SEED)
else:
    from src.data_loader import align, load_features, load_groups, load_labels
    features = load_features(ROOT / "data/raw/features.csv")
    labels   = load_labels(ROOT / "data/raw/labels.csv")
    features, labels = align(features, labels)
    clusters = load_groups(ROOT / "data/raw/genetic_groups.csv", index=features.index)

print(f"{features.shape[0]:,} genomes x {features.shape[1]} determinants")
print(f"{labels.shape[1]} antibiotics, {clusters.nunique()} genetic lineages")
features.head()
"""),
    md("""
## 3. What the data looks like

Two properties drive every modelling decision that follows.

**Class imbalance.** Isolates get sequenced when they cause trouble, so resistance is
over-represented. A model answering "resistant" every time would score well on raw
accuracy — which is why nothing below reports raw accuracy.

**Lineage structure.** Genomes fall into near-identical clonal groups. This is what makes
a random row split dishonest.
"""),
    code("""
summary = pd.DataFrame({
    "tested": labels.notna().sum(),
    "resistant": (labels == 1).sum(),
    "susceptible": (labels == 0).sum(),
})
summary["resistant_%"] = (100 * summary.resistant / summary.tested).round(1)
display(summary)

sizes = clusters.value_counts()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].bar(summary.index, summary["resistant_%"], color="#2a78d6")
axes[0].axhline(50, ls="--", c="grey", lw=1)
axes[0].set_ylabel("% resistant"); axes[0].set_title("Class balance")
axes[0].tick_params(axis="x", rotation=45)
for t in axes[0].get_xticklabels():
    t.set_ha("right")
axes[1].hist(sizes.values, bins=20, color="#1baf7a")
axes[1].set_xlabel("genomes per lineage"); axes[1].set_ylabel("lineages")
axes[1].set_title(f"Lineage structure ({clusters.nunique()} groups)")
plt.tight_layout(); plt.show()
"""),
    md("""
## 4. Split by lineage, not by row

Whole lineages go to one split. If near-identical isolates sit in both train and test,
the score measures recognition of lineages already seen rather than resistance biology.

The calibration set is held out from test as well. A calibrator fitted on test data has
seen its own answers, which makes the Brier score and reliability curve look better than
they are.
"""),
    code("""
from src.utils.clustering import grouped_split, verify_no_leakage

split = grouped_split(clusters, test_frac=0.25, calibration_frac=0.15, seed=SEED)
verify_no_leakage(split, clusters)      # raises if any lineage spans two splits
print("leak-free —", split.summary())

X_train, y_train = features.loc[split.train], labels.loc[split.train]
X_cal,   y_cal   = features.loc[split.calibration], labels.loc[split.calibration]
X_test,  y_test  = features.loc[split.test], labels.loc[split.test]
"""),
    md("""
## 5. Train one model per antibiotic

L1 is a deliberate choice, not a default. It drives most coefficients to zero, leaving a
short list of determinants per drug — and a prediction you can put a gene name next to is
worth more here than a slightly better score you cannot explain.

`class_weight="balanced"` stops the imbalance above from being learned as "answer
resistant".

Rows where the drug was not tested are dropped per drug, which is why each model sees a
different number of genomes.
""" ),
    code("""
def fit_one_drug(drug, X_train, y_train, C=C, seed=SEED):
    \"\"\"L1 logistic regression for a single antibiotic. Returns None if untrainable.\"\"\"
    mask = y_train[drug].notna()
    X, y = X_train[mask], y_train.loc[mask, drug].astype(int)

    # Both classes must be present, or there is nothing to separate.
    if y.nunique() < 2 or len(y) < 30:
        print(f"  skip {drug}: {len(y)} labelled, {y.nunique()} class(es)")
        return None

    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=C,
        class_weight="balanced", max_iter=2000, random_state=seed,
    )
    model.fit(X.to_numpy(), y.to_numpy())
    n_selected = int((model.coef_[0] != 0).sum())
    print(f"  fit  {drug}: {int(y.sum())} resistant / {int((1-y).sum())} susceptible, "
          f"{n_selected} determinants selected")
    return model


print("Training:")
raw_models = {}
for drug in labels.columns:
    m = fit_one_drug(drug, X_train, y_train)
    if m is not None:
        raw_models[drug] = m
print(f"\\n{len(raw_models)} drug models trained")
"""),
    md("""
## 6. Calibrate the probabilities

A raw `predict_proba` is a score, not a probability. Calibration makes the number mean
what it says: among the cases labelled 80%, about 80% should actually be resistant.

Isotonic is flexible but needs data; on a small calibration split it fits noise, so we
fall back to Platt (a logistic fit on the scores) below ~200 samples.

Fitted on `X_cal` — held out from both training and test.
"""),
    code("""
def fit_calibrator(drug, model, X_cal, y_cal):
    mask = y_cal[drug].notna()
    X, y = X_cal[mask], y_cal.loc[mask, drug].astype(int)
    if len(y) == 0 or y.nunique() < 2:
        print(f"  {drug}: no usable calibration data -> identity")
        return None, "identity"

    scores = model.predict_proba(X.to_numpy())[:, 1]

    if len(y) >= 200:
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        cal.fit(scores, y.to_numpy())
        method = "isotonic"
    else:
        # C=1e6 makes this an essentially unregularized 1-D logistic fit, which is what
        # Platt scaling means. sklearn's default C=1.0 would shrink the fit and give
        # different probabilities from the module the app serves.
        cal = LogisticRegression(C=1e6, solver="lbfgs")
        cal.fit(scores.reshape(-1, 1), y.to_numpy())
        method = "platt"
    print(f"  {drug}: {method} on {len(y)} samples")
    return cal, method


def apply_calibrator(cal, method, scores):
    if cal is None:
        return scores
    if method == "isotonic":
        return cal.predict(scores)
    return cal.predict_proba(scores.reshape(-1, 1))[:, 1]


print("Calibrating:")
calibrators = {}
for drug, model in raw_models.items():
    calibrators[drug] = fit_calibrator(drug, model, X_cal, y_cal)
"""),
    md("""
## 7. Decisions, and the two rules that stop false confidence

A probability becomes one of three answers. Both rules below exist because of failures
found while building this — they are not decoration.

**The no-call band.** Between `LOW` and `HIGH` the model returns no-call. Forcing every
case to yes/no manufactures confidence, and a confident wrong answer sends a care team
toward the wrong drug.

**A failure call must cite evidence.** With a cohort that is ~80% resistant, the
regression learns a large positive intercept, and a genome carrying *zero* resistance
determinants gets scored as likely to fail. That number reflects the training prior, not
this isolate. Without a detected determinant the call is downgraded to no-call.
"""),
    code("""
def decide(probability, has_evidence, low=LOW, high=HIGH):
    \"\"\"Map probability + evidence to a decision and a confidence.

    Confidence is None for a no-call: there is no claim to be confident about, and any
    closeness-to-boundary measure would peak exactly where the model knows least.
    \"\"\"
    if probability >= high:
        if not has_evidence:
            return "no-call", None, "no determinant detected — score reflects training prevalence"
        return "likely to fail", probability, ""
    if probability <= low:
        return "likely to work", 1.0 - probability, ""
    return "no-call", None, "probability sits inside the uncertain band"


# Sanity: an empty genome must never assert failure.
for p in (0.10, 0.50, 0.88):
    print(f"p={p:.2f}  no evidence -> {decide(p, has_evidence=False)[0]:15}"
          f"   with evidence -> {decide(p, has_evidence=True)[0]}")
"""),
    md("""
## 8. Evaluate on held-out lineages

Balanced accuracy and PR-AUC rather than raw accuracy, for the imbalance reason above.
Brier score for whether the probabilities are honest. And the pair that matters most for
a decision-support tool: how often it declines, and how accurate it is on the calls it
does make.
"""),
    code("""
def evaluate(drug, model, calibrator, X_test, y_test):
    mask = y_test[drug].notna()
    X, y = X_test[mask], y_test.loc[mask, drug].astype(int).to_numpy()
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None

    cal, method = calibrator
    probability = apply_calibrator(cal, method, model.predict_proba(X.to_numpy())[:, 1])

    evidence = (X.to_numpy() != 0).any(axis=1)
    decisions = [decide(p, e)[0] for p, e in zip(probability, evidence)]
    called = np.array([d != "no-call" for d in decisions])
    predicted = np.array([d == "likely to fail" for d in decisions], dtype=int)

    return {
        "drug": drug,
        "n_test": len(y),
        "balanced_accuracy": balanced_accuracy_score(y, probability >= 0.5),
        "recall_resistant": recall_score(y, probability >= 0.5, pos_label=1, zero_division=0),
        "recall_susceptible": recall_score(y, probability >= 0.5, pos_label=0, zero_division=0),
        "f1": f1_score(y, probability >= 0.5, zero_division=0),
        "auroc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "brier": brier_score_loss(y, probability),
        "no_call_rate": 1.0 - called.mean(),
        "accuracy_on_called": (predicted[called] == y[called]).mean() if called.any() else np.nan,
    }


rows = [evaluate(d, m, calibrators[d], X_test, y_test) for d, m in raw_models.items()]
metrics = pd.DataFrame([r for r in rows if r]).set_index("drug")
metrics.round(3)
"""),
    md("""
### Do the confidence numbers mean anything?

Bin predictions by stated probability, then plot the observed rate in each bin. On the
diagonal means the number shown to a clinician is trustworthy.
"""),
    code("""
def reliability(y_true, probability, n_bins=8):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probability, edges) - 1, 0, n_bins - 1)
    pred, obs = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 5:
            pred.append(probability[m].mean())
            obs.append(y_true[m].mean())
    return np.array(pred), np.array(obs)


drugs = list(raw_models)
cols = min(3, len(drugs)); rows_n = (len(drugs) + cols - 1) // cols
fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.2 * rows_n), squeeze=False)

for ax, drug in zip(axes.ravel(), drugs):
    mask = y_test[drug].notna()
    y = y_test.loc[mask, drug].astype(int).to_numpy()
    cal, method = calibrators[drug]
    p = apply_calibrator(cal, method,
                         raw_models[drug].predict_proba(X_test[mask].to_numpy())[:, 1])
    pred, obs = reliability(y, p)
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    if len(pred):
        ax.plot(pred, obs, "o-", c="#2a78d6")
    ax.set_title(drug, fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("observed")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for ax in axes.ravel()[len(drugs):]:
    ax.axis("off")
plt.tight_layout(); plt.show()
"""),
    md("""
### What each model learned

L1 leaves a short list per drug. Positive weight pushes toward resistance.

Read these as correlations, not mechanisms. A determinant riding the same plasmid as the
real cause picks up weight too — which is exactly why the app labels evidence as either a
*known determinant* or a *statistical association only*, and never presents a coefficient
as proof of biology.
"""),
    code("""
rows = []
for drug, model in raw_models.items():
    coefs = pd.Series(model.coef_[0], index=features.columns)
    for name, value in coefs[coefs != 0].sort_values(key=abs, ascending=False).head(6).items():
        rows.append({"drug": drug, "determinant": name, "coefficient": round(value, 3),
                     "pushes_toward": "resistance" if value > 0 else "susceptibility"})
pd.DataFrame(rows).head(30)
"""),
    md("""
## 9. Save the model for the frontend

The cells above trained `raw_models` and `calibrators`. The Streamlit app expects a
`GenomeFirewall` panel, so we load what the notebook built into that object and save it —
one `joblib` file, exactly where the app looks.

**Then we check the two agree.** If a cell here has drifted from `src/predictor.py`, the
assert fires. Without it, the notebook could quietly ship a model that predicts one thing
while the demo shows another, and nothing would surface the difference until a judge
uploaded a genome.
"""),
    code("""
from src.predictor import DrugModel, GenomeFirewall
from src.utils.calibration import ProbabilityCalibrator

panel = GenomeFirewall(species=SPECIES, C=C, low=LOW, high=HIGH)
panel.feature_names_ = list(features.columns)

for drug, model in raw_models.items():
    dm = DrugModel(drug_name=drug, C=C, low=LOW, high=HIGH)
    dm.model = model                       # the estimator this notebook fitted
    dm.feature_names_ = list(features.columns)

    cal, method = calibrators[drug]
    wrapper = ProbabilityCalibrator(method="auto")
    # The fitted estimator lives in the private `_model`. Setting a differently-named
    # attribute leaves `_model` as None, and transform() then returns raw uncalibrated
    # scores — quietly, with no error.
    wrapper._model = cal
    wrapper.fitted_method_ = method
    dm.calibrator = wrapper
    dm.is_fitted_ = True

    panel.models[drug] = dm

OUT = ROOT / "artifacts" / "genome_firewall.joblib"
OUT.parent.mkdir(parents=True, exist_ok=True)
panel.save(OUT)
print(f"saved -> {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
"""),
    code("""
# Reload from disk exactly as the app does, and confirm it matches this notebook.
reloaded = GenomeFirewall.load(OUT)

probe = {"gene:blaCTX-M-15": 1, "class:BETA-LACTAM": 1}
print("Reloaded model on an ESBL-positive genome:\\n")
for pred in reloaded.predict_genome(probe):
    figure = f"{pred.confidence:.0%}" if pred.is_called else f"p={pred.resistance_probability:.0%}"
    print(f"  {pred.drug:32} {pred.decision:16} {figure:>6}  {pred.evidence_type}")

# The module the app serves must agree with the cells above, drug by drug.
sample = X_test.head(50)
for drug, model in raw_models.items():
    cal, method = calibrators[drug]
    from_notebook = apply_calibrator(cal, method,
                                     model.predict_proba(sample.to_numpy())[:, 1])
    from_module = reloaded.models[drug].predict_proba(sample)
    assert np.allclose(from_notebook, from_module, atol=1e-9), (
        f"{drug}: notebook and src/predictor.py disagree — the app would serve "
        f"different predictions than this notebook reports"
    )
print(f"\\n✅ notebook and src/predictor.py agree on all {len(raw_models)} drugs")
"""),
    md("""
## 10. The frontend

The app reads `artifacts/genome_firewall.joblib` — the file just written. Nothing else to
wire up:

```bash
streamlit run app/streamlit_app.py
```

Re-running section 9 replaces the model the app serves. Reload the page to pick it up.

---

### Where to experiment

- `C` (section 1) — lower selects fewer determinants. Watch the coefficient table.
- `LOW` / `HIGH` — widen the band for a more cautious system; watch `no_call_rate` rise
  and `accuracy_on_called` with it.
- The isotonic/Platt cutoff in section 6.
- `fit_one_drug` — try `penalty="l2"`, or swap in a different classifier entirely. The
  assert in section 9 will tell you if the module can still serve what you built.

**Limits.** One species, the antibiotics above, strictly defensive: this predicts
resistance that already exists. Research prototype — every report needs laboratory
confirmation before it informs treatment.
"""),
]

out = Path(__file__).parent / "modeling.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(nb.cells)} cells)")
