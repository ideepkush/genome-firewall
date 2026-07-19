"""Synthetic AMR cohort for pipeline verification.

The organizer dataset is not published until the event, so this generator produces data
with the same shape and the same statistical hazards: lineage structure (clonal
clusters), class imbalance, missing labels, and resistance driven by a few real
determinants plus noise.

This is scaffolding for testing the pipeline end-to-end. Every reported result in the
submission must come from the real challenge dataset — synthetic numbers prove the code
runs, not that the model works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Determinants named after real AMR genes so the evidence output reads sensibly.
GENE_POOL = {
    "BETA-LACTAM": ["blaTEM-1", "blaCTX-M-15", "blaSHV-12", "blaOXA-48", "blaKPC-2", "blaNDM-1"],
    "QUINOLONE": ["qnrS1", "qnrB19", "aac(6')-Ib-cr", "oqxA"],
    "AMINOGLYCOSIDE": ["aac(3)-IIa", "aph(3'')-Ib", "aph(6)-Id", "armA"],
    "TETRACYCLINE": ["tet(A)", "tet(B)", "tet(M)"],
    "SULFONAMIDE": ["sul1", "sul2"],
    "TRIMETHOPRIM": ["dfrA17", "dfrA1"],
    "POLYMYXIN": ["mcr-1"],
}

POINT_POOL = {
    "QUINOLONE": ["gyrA_S83L", "gyrA_D87N", "parC_S80I"],
    "POLYMYXIN": ["pmrB_L10P"],
}

# Which determinants actually drive resistance for each drug, and how strongly.
DRUG_MECHANISMS = {
    "ampicillin": {"class": "BETA-LACTAM", "drivers": {"blaTEM-1": 0.92, "blaCTX-M-15": 0.95, "blaSHV-12": 0.88}},
    "ceftriaxone": {"class": "BETA-LACTAM", "drivers": {"blaCTX-M-15": 0.94, "blaKPC-2": 0.90, "blaNDM-1": 0.96}},
    "meropenem": {"class": "BETA-LACTAM", "drivers": {"blaKPC-2": 0.93, "blaNDM-1": 0.97, "blaOXA-48": 0.85}},
    "ciprofloxacin": {"class": "QUINOLONE", "drivers": {"gyrA_S83L": 0.80, "gyrA_D87N": 0.75, "parC_S80I": 0.70, "qnrS1": 0.45}},
    "gentamicin": {"class": "AMINOGLYCOSIDE", "drivers": {"aac(3)-IIa": 0.90, "armA": 0.96}},
    "trimethoprim-sulfamethoxazole": {"class": "SULFONAMIDE", "drivers": {"sul1": 0.70, "sul2": 0.68, "dfrA17": 0.85}},
}


def generate_cohort(
    n_genomes: int = 1200,
    n_clusters: int = 40,
    drugs: list[str] | None = None,
    missing_label_rate: float = 0.15,
    background_resistance: float = 0.04,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate (features, labels, clusters).

    Determinant carriage is drawn per *cluster*, not per genome, which is what creates
    the leakage hazard the pipeline must defend against: genomes within a cluster look
    nearly identical, so a random split would let the model memorize them.

    Returns:
        features: genomes x binary determinant features (gene:/point:/class: prefixed)
        labels:   genomes x drugs, 1 = resistant, 0 = susceptible, NaN = not tested
        clusters: genome_id -> cluster_id
    """
    rng = np.random.default_rng(seed)
    drugs = drugs or list(DRUG_MECHANISMS)

    all_genes = [g for genes in GENE_POOL.values() for g in genes]
    all_points = [p for points in POINT_POOL.values() for p in points]

    gene_to_class = {g: cls for cls, genes in GENE_POOL.items() for g in genes}
    gene_to_class.update({p: cls for cls, points in POINT_POOL.items() for p in points})

    # Each cluster gets a carriage probability per determinant: mostly near 0 or near 1,
    # so members of a cluster share a profile the way a real clonal lineage does.
    cluster_profiles = {
        cid: {
            det: float(rng.beta(0.25, 0.25))
            for det in all_genes + all_points
        }
        for cid in range(n_clusters)
    }

    cluster_assignment = rng.integers(0, n_clusters, size=n_genomes)
    genome_ids = [f"GEN{idx:05d}" for idx in range(n_genomes)]

    feature_rows, label_rows = [], []
    for genome_id, cluster_id in zip(genome_ids, cluster_assignment):
        profile = cluster_profiles[int(cluster_id)]
        carried = {det: int(rng.random() < prob) for det, prob in profile.items()}

        features: dict[str, int] = {}
        for det, present in carried.items():
            if not present:
                continue
            prefix = "point" if det in all_points else "gene"
            features[f"{prefix}:{det}"] = 1
            features[f"class:{gene_to_class[det]}"] = 1

        labels: dict[str, float] = {}
        for drug in drugs:
            mechanism = DRUG_MECHANISMS[drug]
            # Resistance probability is the noisy-OR of the carried drivers, plus a
            # small background rate for mechanisms outside the annotated feature set.
            p_susceptible = 1.0 - background_resistance
            for driver, strength in mechanism["drivers"].items():
                if carried.get(driver, 0):
                    p_susceptible *= 1.0 - strength

            resistant = int(rng.random() < (1.0 - p_susceptible))
            labels[drug] = np.nan if rng.random() < missing_label_rate else float(resistant)

        feature_rows.append(features)
        label_rows.append(labels)

    features_df = pd.DataFrame(feature_rows, index=genome_ids).fillna(0).astype(int)
    features_df.index.name = "genome_id"
    features_df = features_df.sort_index(axis=1)

    labels_df = pd.DataFrame(label_rows, index=genome_ids)
    labels_df.index.name = "genome_id"

    clusters = pd.Series(cluster_assignment, index=genome_ids, name="cluster_id")
    clusters.index.name = "genome_id"

    return features_df, labels_df, clusters
