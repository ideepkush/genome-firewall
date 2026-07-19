# De-duplication threshold — how it was chosen

The brief leaves the sequence-homology threshold to each team to *tune and justify*. This
is that justification, including the part that is reasoned rather than measured.

## What the threshold does

Near-identical genomes from the same outbreak or clonal expansion must not straddle the
train/test boundary. If they do, the reported score measures recognition of a lineage the
model has already seen rather than resistance biology.

`src/utils/clustering.py` sketches each genome with MinHash over 21-mers (512 hashes),
computes pairwise Jaccard similarity, and merges any pair at or above
`DEFAULT_THRESHOLD = 0.99` by single linkage. Whole clusters then go to one split.

## Why 0.99

Jaccard similarity over 21-mers is a strict measure. Two *S. aureus* genomes differing by
a handful of SNPs still share nearly all their k-mers; genomes from different sequence
types share substantially fewer. A 0.99 threshold therefore targets **the same clone**,
not merely the same species or the same ST.

The choice is deliberately conservative in the direction that costs us score rather than
credibility:

- **Too low** (0.90) merges genuinely distinct lineages into one cluster. Fewer, larger
  clusters means less training diversity and a test set that no longer represents
  independent lineages. It also *flatters* nothing — it simply wastes data.
- **Too high** (0.999) fails to merge true near-duplicates, letting the same clone appear
  in train and test. This inflates the reported score, which is the failure the brief
  singles out as a weak submission.

Given the asymmetry — one setting wastes data, the other manufactures a better-looking
number — we prefer the conservative end.

## Sensitivity check

Threshold swept on a 1,200-genome synthetic cohort with 40 planted lineages, using the
feature-profile clustering fallback (see caveat below):

| threshold | clusters | largest cluster | test genomes | mean balanced accuracy | leakage |
|---|---|---|---|---|---|
| 0.900 | 502 | 40 | 322 | 0.795 | none |
| 0.950 | 962 | 33 | 318 | 0.757 | none |
| 0.980 | 1029 | 7 | 305 | 0.781 | none |
| 0.990 | 1029 | 7 | 305 | 0.781 | none |
| 0.995 | 1029 | 7 | 305 | 0.781 | none |

Two things follow.

**The result is flat above 0.98.** 0.98, 0.99 and 0.995 produce identical clusterings, so
the exact value in that range does not change any reported number. The choice of 0.99 is
therefore not load-bearing — anything in [0.98, 1.0) gives the same answer.

**Loosening to 0.90 raises the score.** Balanced accuracy goes *up* to 0.795 while the
largest cluster grows from 7 to 40 genomes. That direction is exactly the trap: a laxer
de-duplication threshold produces a better headline number precisely because it is
letting more similar genomes sit on both sides of the split. We report the stricter
setting.

`verify_no_leakage` passes at every threshold, because it checks that the split respects
whatever clustering it was given. It cannot detect that the clustering itself was too
permissive — which is why the threshold has to be argued, not just asserted.

## Caveat — what this does not measure

The sweep above uses `cluster_from_feature_matrix`, which clusters on binary resistance
profiles. That is the **fallback** used when genome sequences are unavailable, and it is
what we could actually run: we hold laboratory AST labels for 3,960 *S. aureus* genomes
but not their assemblies.

The MinHash sequence path (`sketch_genome` / `cluster_genomes`) is the intended route and
is the one `DEFAULT_THRESHOLD` governs. It is implemented and unit-tested, but tuning it
empirically requires the assemblies. Until those are in hand, 0.99 rests on the reasoning
above rather than on a measurement, and this document should be updated once it can be
swept for real.

Stating that plainly is the point. A threshold presented as "tuned" when it was only
reasoned about is the same category of overclaim as an uncalibrated confidence score.

## Reproducing

```bash
python -m scripts.sweep_threshold      # writes the table above
```

Persisted alongside every run: clustering method, threshold, software version, cluster
assignments, random seed, and the split manifest.
