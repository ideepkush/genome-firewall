"""Sequence-homology de-duplication and grouped train/calibration/test splitting.

This module exists to prevent the single most common way genomic AMR models produce
inflated scores: near-identical isolates from the same outbreak landing in both the
training and test sets. A random row split on such data measures memorization, not
resistance biology.

We cluster genomes by k-mer (MinHash) similarity, then split at the *cluster* level so
no cluster is ever spread across two sets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_K = 21
DEFAULT_SKETCH_SIZE = 512
DEFAULT_THRESHOLD = 0.99


def _kmer_hashes(sequence: str, k: int, sketch_size: int) -> np.ndarray:
    """Bottom-k MinHash sketch of a sequence.

    Keeping the smallest `sketch_size` k-mer hashes gives an unbiased estimate of
    Jaccard similarity at a fixed memory cost per genome.
    """
    sequence = sequence.upper()
    if len(sequence) < k:
        return np.array([], dtype=np.uint64)

    hashes = {
        int.from_bytes(
            hashlib.blake2b(sequence[i : i + k].encode(), digest_size=8).digest(),
            "little",
        )
        for i in range(len(sequence) - k + 1)
    }
    return np.sort(np.fromiter(hashes, dtype=np.uint64))[:sketch_size]


def sketch_genome(sequences: list[str], k: int = DEFAULT_K, sketch_size: int = DEFAULT_SKETCH_SIZE) -> np.ndarray:
    """Sketch a whole assembly by pooling the sketches of its contigs."""
    pooled: set[int] = set()
    for seq in sequences:
        pooled.update(int(h) for h in _kmer_hashes(seq, k, sketch_size))
    return np.sort(np.fromiter(pooled, dtype=np.uint64))[:sketch_size]


def jaccard(sketch_a: np.ndarray, sketch_b: np.ndarray) -> float:
    """Estimated Jaccard similarity between two MinHash sketches."""
    if sketch_a.size == 0 or sketch_b.size == 0:
        return 0.0
    intersection = np.intersect1d(sketch_a, sketch_b, assume_unique=True).size
    union = np.union1d(sketch_a, sketch_b).size
    return intersection / union if union else 0.0


def cluster_by_similarity(
    sketches: dict[str, np.ndarray], threshold: float = DEFAULT_THRESHOLD
) -> pd.Series:
    """Single-linkage clustering of genomes above a Jaccard similarity threshold.

    Returns a Series mapping genome_id -> cluster_id. Single linkage is the
    conservative choice here: it errs toward merging, which keeps related isolates
    together rather than risking a leak across the split.
    """
    genome_ids = list(sketches)
    parent = {gid: gid for gid in genome_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, gid_a in enumerate(genome_ids):
        for gid_b in genome_ids[i + 1 :]:
            if jaccard(sketches[gid_a], sketches[gid_b]) >= threshold:
                union(gid_a, gid_b)

    roots = {gid: find(gid) for gid in genome_ids}
    labels = {root: idx for idx, root in enumerate(sorted(set(roots.values())))}
    return pd.Series({gid: labels[root] for gid, root in roots.items()}, name="cluster_id")


def cluster_from_feature_matrix(
    matrix: pd.DataFrame, threshold: float = 0.98
) -> pd.Series:
    """Fallback clustering when raw sequences are unavailable.

    Groups genomes by their resistance-gene profile using Jaccard similarity over the
    binary feature vectors. This is weaker than k-mer clustering — two unrelated
    isolates can share a resistance profile — so prefer the organizer's genetic groups
    when the dataset provides them.
    """
    ids = list(matrix.index)
    values = matrix.to_numpy(dtype=bool)
    parent = {gid: gid for gid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = values[i], values[j]
            union_count = np.logical_or(a, b).sum()
            if union_count == 0:
                continue
            if np.logical_and(a, b).sum() / union_count >= threshold:
                ra, rb = find(ids[i]), find(ids[j])
                if ra != rb:
                    parent[rb] = ra

    roots = {gid: find(gid) for gid in ids}
    labels = {root: idx for idx, root in enumerate(sorted(set(roots.values())))}
    return pd.Series({gid: labels[root] for gid, root in roots.items()}, name="cluster_id")


@dataclass
class GroupedSplit:
    """Genome ids for each stage, guaranteed cluster-disjoint."""

    train: list[str]
    calibration: list[str]
    test: list[str]

    def summary(self) -> str:
        total = len(self.train) + len(self.calibration) + len(self.test)
        return (
            f"train={len(self.train)} calibration={len(self.calibration)} "
            f"test={len(self.test)} (total={total})"
        )


def grouped_split(
    clusters: pd.Series,
    test_frac: float = 0.25,
    calibration_frac: float = 0.15,
    seed: int = 42,
) -> GroupedSplit:
    """Split whole clusters into train / calibration / test.

    The calibration set is held out separately from test because fitting the
    probability calibrator on test data would make the reported Brier score and
    reliability curve optimistic — the calibrator would have seen its own answers.
    """
    rng = np.random.default_rng(seed)
    cluster_ids = np.array(sorted(clusters.unique()))
    rng.shuffle(cluster_ids)

    n_test = max(1, int(round(len(cluster_ids) * test_frac)))
    n_cal = max(1, int(round(len(cluster_ids) * calibration_frac)))

    test_clusters = set(cluster_ids[:n_test])
    cal_clusters = set(cluster_ids[n_test : n_test + n_cal])

    train, calibration, test = [], [], []
    for genome_id, cluster_id in clusters.items():
        if cluster_id in test_clusters:
            test.append(genome_id)
        elif cluster_id in cal_clusters:
            calibration.append(genome_id)
        else:
            train.append(genome_id)

    return GroupedSplit(train=sorted(train), calibration=sorted(calibration), test=sorted(test))


def verify_no_leakage(split: GroupedSplit, clusters: pd.Series) -> None:
    """Raise if any cluster appears in more than one set. Call this before training."""
    sets = {
        "train": {clusters[g] for g in split.train},
        "calibration": {clusters[g] for g in split.calibration},
        "test": {clusters[g] for g in split.test},
    }
    for a, b in (("train", "test"), ("train", "calibration"), ("calibration", "test")):
        shared = sets[a] & sets[b]
        if shared:
            raise AssertionError(
                f"Cluster leakage between {a} and {b}: {sorted(shared)[:5]} "
                f"({len(shared)} clusters). Metrics from this split would be inflated."
            )
