# Module 01 — feature output specification

`feature_schema_version: 1.0.0`

The brief requires a documented, repeatable path from an assembled FASTA to model
features, plus a specification of the output format. This is that specification. The same
builder runs in training and inference; a mismatch between the two is the failure mode
this document exists to prevent.

## Contract

`featurize_fasta(path, organism=None)` returns `(genome_id, features, qc, hits)`.

`features` is a `dict[str, int]` — determinant name to presence, always 0 or 1. Encoded
into a genome × determinant matrix, absence is `0`, never `NaN`.

## Naming

Three prefixed families, one per line of evidence:

| Prefix | Source | Example |
|---|---|---|
| `gene:` | acquired resistance gene | `gene:mecA`, `gene:ermC`, `gene:tet(K)` |
| `point:` | resistance-associated point mutation | `point:gyrA_S84L`, `point:grlA_S80F` |
| `class:` | drug-class rollup of any hit | `class:BETA-LACTAM`, `class:MACROLIDE` |

Rules:

- The name after the prefix is AMRFinderPlus's own symbol, **verbatim** — including
  parentheses (`tet(K)`) and case (`mecA`, not `meca`). Renaming breaks the join between a
  detected determinant and the drug knowledge base.
- `class:` values are AMRFinderPlus `Class` values, uppercase. These are the tool's
  vocabulary, not ours: `sul1` is reported under `SULFONAMIDE`, and a knowledge base
  calling the same thing `FOLATE-PATHWAY-ANTAGONIST` silently fails to match.
- Point mutations are `{gene}_{ref}{position}{alt}` as AMRFinderPlus emits them.
- A hit with an empty symbol is dropped, not encoded as an empty name.

## Which hits are kept

- Element types `AMR` and `POINT`. Virulence and stress-response hits are annotated by the
  tool but are out of scope and excluded.
- Every retained hit contributes its `gene:`/`point:` feature **and** its `class:` rollup.
  One hit therefore produces at least two features.

## Prevalence filter

Determinants present in fewer than **3** genomes across the training cohort are dropped
(`--min-prevalence`, default 3). A determinant seen once or twice cannot generalize; it
can only let the model memorize the isolate carrying it.

Applied at matrix-build time using training data only. The fitted feature list is frozen
into the model bundle, so inference uses the training vocabulary.

## Schema stability between training and inference

At inference the feature vector is reindexed onto the model's stored `feature_names_`:

- determinant in the model's vocabulary, detected in this genome → `1`
- determinant in the vocabulary, not detected → `0`
- determinant detected but **not** in the vocabulary → logged and ignored

The third case is unavoidable — a new genome may carry a gene absent from training — but
it is a signal worth watching. A high unseen-feature rate means the isolate is unlike
anything the model was fitted on, which is grounds for a no-call rather than a confident
answer.

## Absence is not evidence of susceptibility

`0` means *this determinant was not detected*. It does not mean the genome is susceptible.
A poor assembly, a truncated contig, or a failed annotation all produce zeros that look
identical to genuine absence.

This is why the QC record travels with the features, and why the decision policy treats
"no determinant found" as `NO_KNOWN_RESISTANCE_SIGNAL` rather than as evidence for
`LIKELY_TO_WORK`.

## QC record

`qc` accompanies every feature vector and carries: total assembly length, contig count,
N50, ambiguous-base fraction, and the flags raised (`assembly_too_short`,
`too_many_contigs`, `high_ambiguous_bases`). Flags do not block annotation; they are
surfaced in the report and are inputs to the no-call decision.

## Annotation provenance

Pinned and recorded with every run:

```
tool:        AMRFinderPlus
version:     4.2.7
database:    2026-05-15.1
```

The parser accepts both v3 and v4 column spellings and **raises** on an unrecognised
schema rather than returning an empty hit list. A silently empty result is
indistinguishable from a clean genome, which is the most dangerous failure this module can
produce — v4's column rename caused exactly that, and a genome carrying five resistance
genes was reported as clean until it was caught.

## Worked example

Input: *S. aureus* assembly carrying `mecA` and `ermC`.

```json
{
  "gene:mecA": 1,
  "gene:ermC": 1,
  "class:BETA-LACTAM": 1,
  "class:MACROLIDE": 1
}
```

Every other determinant in the training vocabulary is `0`. Downstream, `mecA` maps to
cefoxitin as a known mechanism (PBP2a — this is MRSA) and `ermC` to erythromycin.
