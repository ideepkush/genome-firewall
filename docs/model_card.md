# Model card — Genome Firewall

| | |
|---|---|
| **Version** | 0.1.0 |
| **Species** | *Staphylococcus aureus* — this species only |
| **Antibiotics** | cefoxitin, ciprofloxacin, erythromycin, tetracycline |
| **Task** | Per-antibiotic phenotypic resistance prediction from an assembled genome |
| **Output** | `LIKELY_TO_FAIL` / `LIKELY_TO_WORK` / `NO_CALL` + calibrated confidence + evidence |
| **Model** | One L1-penalized logistic regression per antibiotic |
| **Annotation** | AMRFinderPlus 4.2.7, database 2026-05-15.1 |
| **Feature schema** | 1.0.0 (see `docs/feature_schema.md`) |
| **Status** | ⚠️ **Research prototype trained on synthetic data — not validated on real genomes** |

## ⚠️ Read this before any number below

**The metrics in this card come from a synthetic cohort.** They verify that the pipeline
runs correctly and that its honesty properties hold. They say **nothing** about
real-world performance.

We hold laboratory AST labels for 3,960 real *S. aureus* genomes (BV-BRC, all
`Laboratory Method`, no computational predictions) but not the corresponding assemblies,
so features cannot be built and the model has not been fitted on real data. Any real
evaluation is pending that.

## Intended use

**Intended.** Research decision support for clinical microbiologists and infectious-disease
clinicians while laboratory susceptibility testing is still running, and for public-health
teams tracking which resistance mechanisms are circulating.

**Not intended.** Making a treatment decision. Replacing susceptibility testing. Any
species other than *S. aureus*, any antibiotic outside the four above, raw reads, mixed
samples, species identification, or genome assembly. The system starts after isolation,
sequencing, assembly and species identification are complete.

**Never.** Designing, modifying, or optimizing an organism. The codebase contains no
generative capability, by construction.

## Inputs

One quality-checked, assembled *S. aureus* genome in FASTA. Encoded to binary determinant
features via AMRFinderPlus — `gene:`, `point:`, and `class:` families, specified in
`docs/feature_schema.md`.

## Performance (synthetic cohort, held-out lineages)

1,200 genomes, 40 planted lineages, split by lineage with `verify_no_leakage` enforced
before training.

| drug | n | balanced acc | recall (R) | recall (S) | PR-AUC | Brier | no-call |
|---|---|---|---|---|---|---|---|
| cefoxitin | 247 | 0.955 | 0.988 | 0.921 | 0.970 | 0.033 | 0.000 |
| tetracycline | 259 | 0.891 | 0.930 | 0.851 | 0.950 | 0.082 | 0.000 |
| erythromycin | 250 | 0.684 | 1.000 | 0.368 | 0.997 | 0.037 | 0.060 |
| ciprofloxacin | 249 | 0.617 | 0.968 | 0.267 | 0.966 | 0.098 | 0.024 |

### Read the two weak rows

Erythromycin and ciprofloxacin have high resistant recall and **poor susceptible recall**
(0.37, 0.27). The cohort gave erythromycin 595 resistant against 26 susceptible examples
— too few to learn what susceptibility looks like, so the model defaults toward
resistance.

Balanced accuracy near 0.62–0.68 is the honest summary. Note that PR-AUC stays high
(0.97+) precisely *because* of the imbalance: it is a flattering metric here and should
not be quoted alone.

Real *S. aureus* collections are skewed the same way, so this is a limitation to expect
rather than an artefact of the generator.

## Behaviour that is enforced, not incidental

- **A failure call must cite a determinant.** With an ~80% resistant cohort the regression
  learns a large positive intercept and scores a determinant-free genome as likely to
  fail — reporting the training prior as though it were a finding about this isolate.
  Such calls are downgraded to `NO_CALL`. (`test_empty_genome_never_asserts_failure`)
- **A no-call carries no confidence figure.** There is no claim to be confident about, and
  any closeness-to-boundary measure peaks exactly where the model knows least. The raw
  resistance probability is shown instead.
  (`test_no_call_reports_no_confidence`)
- **Known mechanisms are separated from statistical associations.** A non-zero coefficient
  is never presented as biological cause.
- **A deterministic target gate runs before the model**, so absence of resistance markers
  is never by itself treated as evidence a drug will work.

## Evidence categories

| Category | Meaning |
|---|---|
| `KNOWN_MECHANISM` | A curated resistance gene or mutation for this drug class was detected |
| `STATISTICAL_ASSOCIATION` | The score is driven by correlated features; causality not claimed |
| `NO_KNOWN_RESISTANCE_SIGNAL` | Nothing detected — absence of evidence, not evidence of susceptibility |
| `INTRINSIC` | The target gate fired; the species lacks a susceptible target |

## Known limitations

- **Not fitted on real genomes.** The headline limitation.
- **Susceptible-class performance is weak** for erythromycin and ciprofloxacin.
- **The de-duplication threshold is reasoned, not measured.** 0.99 MinHash Jaccard;
  empirical tuning needs assemblies we do not have. See `docs/thresholds.md`.
- **No out-of-distribution detection.** A genome unlike anything in training is scored
  normally rather than forced to no-call.
- **Ciprofloxacin resistance is chromosomal** (stepwise `gyrA`/`grlA` mutation), which is
  harder to capture from presence/absence features than an acquired gene like `mecA`.
- **Single annotation tool.** AMRFinderPlus only; no ResFinder/CARD cross-check.
- **`blaZ` is a deliberate confounder** in the synthetic data — common in *S. aureus*,
  correlated with resistance, but not a cause of cefoxitin resistance. The model correctly
  excludes it, and that behaviour should be re-verified on real data.

## Ethical and safety notes

Strictly defensive: predicts and explains resistance that already exists. Every report
carries a mandatory laboratory-confirmation warning, and the covered species and drug list
is stated in the interface so out-of-scope requests are refused rather than answered
poorly.

A confident wrong result points a care team toward the wrong drug. That asymmetry is why
the system prefers abstention over coverage.

## Reproducing

```bash
python train.py --synthetic
pytest tests/ -q          # 34 tests
```

Recorded per run: dataset version, feature schema version, AMRFinderPlus and database
version, random seed, thresholds, and split manifest.
