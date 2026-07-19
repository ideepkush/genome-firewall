# Hack-Nation submission draft — Genome Firewall

Deadline: **Jul 19, 9:00 AM ET**. Copy the sections below into the form fields.

> ⚠️ **Read the "Numbers" note at the bottom before pasting any metric.** The figures in
> `artifacts/` come from synthetic data. Presenting them as real performance would be
> exactly the inflated claim this challenge penalizes.

---

## Challenge

**Challenge 6 — Genome Firewall: An AI Defense System Against Superbugs**

---

## Project Title

`Genome Firewall — evidence-first antibiotic resistance prediction`

---

## Short Description

Predicts which antibiotics are likely to fail from an assembled bacterial genome, days
before standard lab results arrive. Every call cites the specific resistance gene behind
it, confidence is calibrated against measured outcomes, and the system returns an
explicit no-call rather than guessing when the evidence is weak.

---

## 1. Problem & Challenge

Antibiotic-resistant infections are associated with over 4.7 million deaths a year, and
more than a million people die because the drugs they were given no longer work.

The bottleneck is not that resistance exists — it is timing. Standard susceptibility
testing takes one to three days. During that window a clinician has to guess, and every
ineffective course costs the patient time while giving resistant bacteria another chance
to spread. Broad-spectrum cover buys safety at the price of accelerating the very
resistance that caused the problem.

Much of the answer is already in the isolate's DNA. Sequencing is fast and cheap. What is
missing is a system that turns an assembled genome into a prediction a clinician can
actually act on — one that shows its evidence, states honest confidence, and admits when
it does not know.

---

## 2. Target Audience

**Primary: clinical microbiologists and infectious-disease clinicians** deciding
empiric therapy while susceptibility testing is still running. They need a defensible
answer, not a black-box score — something they can check against the genome and justify
in a chart review.

**Secondary: public-health and AMR surveillance teams** tracking which resistance
mechanisms are spreading in which regions, earlier than phenotypic reporting allows.

Neither user is served by a tool that is confidently wrong. A wrong referral is not a
failed query; it is a patient on the wrong drug for another 48 hours.

---

## 3. Solution & Core Features

Genome Firewall takes a quality-checked assembly and returns a per-antibiotic report:
**likely to fail / likely to work / no-call**, with calibrated confidence and the
determinant behind each call.

Three modules:

**Genome Reader** — runs AMRFinderPlus over the assembly and encodes every hit as a named
binary feature (`gene:blaCTX-M-15`, `point:gyrA_S83L`, `class:BETA-LACTAM`). Because every
feature is a named determinant, every prediction traces back to something a
microbiologist can verify.

**Predictor** — one L1-penalized logistic regression per antibiotic, wrapped in two
layers. A deterministic target gate runs *before* the model and rules out drugs the
species is intrinsically resistant to — vancomycin cannot cross a gram-negative outer
membrane, and that is biology, not something to learn from data. Probability calibration
is fitted on a split disjoint from both training and test.

**Decision Report** — a Streamlit app showing each call, its confidence, and which of
four evidence categories it rests on: a known determinant, a statistical association
only, no signal found, or intrinsic resistance.

---

## 4. Unique Selling Proposition (USP)

**A prediction that cannot cite evidence is not allowed to assert failure.**

Resistance-labelled genome collections are heavily skewed toward resistant isolates,
because isolates get sequenced when they cause trouble. A model trained on such a cohort
learns a large positive intercept — and will report a genome carrying *zero* resistance
genes as "likely to fail" with high confidence. It is reporting its training prior as
though it were a finding about that patient's isolate.

We found this in our own model and built a coherence rule against it: a `likely to fail`
call must be backed by at least one detected determinant. Without one, the call is
downgraded to `no-call` and the report says exactly why.

The same principle runs through the rest of the system:

- **A no-call carries no confidence figure.** There is no claim to be confident about.
  We show the raw resistance probability instead.
- **Splits are by genetic lineage, never by row.** Near-identical isolates in both train
  and test measure memorization, not biology. A leakage check runs before training.
- **Known mechanisms are kept distinct from statistical associations.** A non-zero
  coefficient is not proof of causation, and the report never presents one as such.
- **Generalization is reported per lineage, worst case included** — not just an average
  that hides lineage-specific collapse.

Most systems optimize a headline number. This one optimizes being trustworthy at 3am.

---

## 5. Implementation & Technology

**Stack:** Python, scikit-learn, BioPython, AMRFinderPlus (NCBI, public domain),
Streamlit, pandas/numpy, matplotlib.

**Pipeline:** assembled FASTA → AMRFinderPlus annotation → binary determinant features →
lineage-aware split → per-drug L1 logistic regression → isotonic/Platt calibration →
gated three-way decision.

**Deliberate engineering choices:**

- *L1 regularization* zeroes most coefficients, leaving a short, inspectable list of
  determinants per drug. Interpretability is a requirement here, not a nice-to-have.
- *MinHash k-mer sketching* clusters genomes by similarity so whole lineages stay inside
  one split; `verify_no_leakage` raises before training rather than after.
- *A separate calibration split* — fitting the calibrator on test data would make the
  Brier score and reliability curve optimistic, since the calibrator would have seen its
  own answers.
- *Isotonic with a Platt fallback* on small calibration sets, where isotonic would fit
  noise.
- *Schema-tolerant parsing.* AMRFinderPlus v4 renamed most output columns. Our v3-era
  parser matched nothing and silently reported a genome carrying five resistance genes as
  completely clean — silence that reads as good news. The parser now accepts both schemas
  and raises on an unrecognized one rather than returning an empty result.

**Verification:** 29 tests covering leakage, calibration, the target gate, evidence
attribution, and the no-call rules. Verified against AMRFinderPlus 4.2.7, database
2026-05-15.1.

---

## 6. Results & Impact

<!-- FILL FROM A REAL RUN — see "Numbers" note below. Structure to follow: -->

Evaluated on held-out genetic clusters the model never saw during training:

- Balanced accuracy, recall on resistant and susceptible cases reported separately
- PR-AUC per drug (the meaningful metric under this class imbalance)
- Brier score and expected calibration error, with reliability curves
- No-call rate alongside accuracy on the calls actually made
- Per-lineage breakdown including the worst-performing cluster

The per-lineage table is the one we would ask a judge to read. Aggregate scores look
strong while individual held-out clusters can sit at chance — that gap is what a headline
average hides, and it is the honest description of where this system can and cannot be
trusted.

**Impact.** Getting the right antibiotic to a patient a day or two earlier, and reducing
unnecessary broad-spectrum use while the lab catches up. Both matter: one saves the
patient in front of you, the other preserves the drugs that still work.

**Scope and limits, stated plainly.** One species, a fixed antibiotic panel, strictly
defensive — the system predicts resistance that already exists and has no generative
capability. It is a research prototype, not a diagnostic device, and every report
requires laboratory confirmation before it informs treatment.

---

## Most Fun Moment

<!-- Yours to write — it should be genuine. If the debugging session is what stands out: -->

Installing AMRFinderPlus and finally running it on a genome we had spiked with five real
resistance genes — and watching our pipeline report a perfectly clean isolate. The tool
had found all five at 100% identity. Our parser was reading v3 column names against v4
output, so every hit fell on the floor.

The fun part was what it taught us: the most dangerous bug in a medical tool is not one
that crashes. It is the one that returns silence and lets you read it as good news. We
spent the rest of the night hunting for other places the system could be quietly
confident, and found two more.

---

## Technologies / Tags

`Python` `scikit-learn` `BioPython` `AMRFinderPlus` `Streamlit` `Bioinformatics`
`Antimicrobial Resistance` `Genomics` `Model Calibration` `Interpretable ML`
`Public Health` `Biosecurity`

---

## ⚠️ Numbers — read before pasting

`artifacts/metrics_per_drug.csv` currently holds results from **synthetic data**
(`train.py --synthetic`). Those figures — mean balanced accuracy 0.83, PR-AUC 0.97 —
verify the pipeline is correct and say **nothing** about real performance.

Do not put them in section 6 as results.

Before submitting, either:

1. **Run on the real challenge dataset** and paste those numbers, or
2. **State honestly** that the pipeline is validated end-to-end and results on the
   official dataset are pending, describing the metrics you report rather than inventing
   values.

Option 2 costs nothing with these judges. Option 1 with fabricated numbers would be
caught, and this challenge explicitly penalizes inflated claims.

---

## Still missing before you can submit

| Item | Status | Note |
|---|---|---|
| Live project demo link | ❌ **blocker** | Currently localhost only — see below |
| GitHub repository URL | ❌ | Repo not yet pushed |
| Team picture | ❌ | Landscape, faces visible |
| Demo video (60s) | ❌ | UI/UX and product flow |
| Tech video (60s) | ❌ | Stack, architecture, implementation |
| Team video (60s) | ❌ | Who built it, roles |
| Most fun moment | ✏️ | Draft above — make it yours |
| Sections 1–5 | ✅ | Ready to paste |
| Section 6 | ⚠️ | Needs real numbers or an honest statement |

**Deployment is the critical path.** AMRFinderPlus needs conda, which Streamlit Community
Cloud cannot install. Two options:

- **Fast (~20 min):** deploy to Streamlit Community Cloud with the manual-entry tab plus a
  precomputed demo genome, so the full report renders without the binary present.
- **Complete (~1–2 hr):** HuggingFace Spaces with a Docker image carrying conda +
  AMRFinderPlus, so live FASTA upload works in the deployed app.

Given the remaining time and three videos still to record, the fast path is the safer
call. Record the demo video against your local instance, where live annotation works.
