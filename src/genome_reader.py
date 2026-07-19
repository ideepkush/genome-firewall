"""Module 01 — Genome Reader: assembled FASTA -> model features.

The pipeline is deliberately simple and auditable: every feature is the presence or
absence of a named resistance determinant, so any downstream prediction can be traced
back to the exact gene or point mutation that produced it.

Two entry points:
  * `featurize_fasta`      — one genome, runs AMRFinderPlus
  * `build_feature_matrix` — a directory of genomes, or a precomputed AMRFinder TSV
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from .utils import amrfinder
from .utils.amrfinder import AMRHit

FEATURE_SCHEMA_VERSION = "1.0"


@dataclass
class GenomeQC:
    """Assembly statistics used to flag genomes we should not trust."""

    genome_id: str
    n_contigs: int
    total_length: int
    longest_contig: int
    gc_content: float
    n_amr_genes: int

    def flags(self, min_length: int = 1_000_000, max_contigs: int = 1000) -> list[str]:
        problems = []
        if self.total_length < min_length:
            problems.append(f"assembly_too_short({self.total_length:,}bp)")
        if self.n_contigs > max_contigs:
            problems.append(f"too_fragmented({self.n_contigs}_contigs)")
        if self.n_amr_genes == 0:
            problems.append("no_amr_genes_detected")
        return problems


def read_fasta_stats(fasta_path: str | Path) -> tuple[str, dict]:
    """Return (genome_id, assembly statistics) for one FASTA file."""
    fasta_path = Path(fasta_path)
    lengths, gc_count, total = [], 0, 0

    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq = str(record.seq).upper()
        lengths.append(len(seq))
        gc_count += seq.count("G") + seq.count("C")
        total += len(seq)

    if not lengths:
        raise ValueError(f"No sequences found in {fasta_path}")

    stats = {
        "n_contigs": len(lengths),
        "total_length": total,
        "longest_contig": max(lengths),
        "gc_content": round(gc_count / total, 4) if total else 0.0,
    }
    return fasta_path.stem, stats


def hits_to_features(hits: list[AMRHit]) -> dict[str, int]:
    """Collapse AMR hits into a binary feature dict.

    Three feature families, all interpretable:
      gene:<symbol>      acquired resistance gene present
      point:<symbol>     resistance-conferring point mutation present
      class:<drug_class> any determinant against this drug class present
    """
    features: dict[str, int] = {}
    for hit in hits:
        if not hit.gene_symbol:
            continue
        prefix = "point" if hit.is_point_mutation else "gene"
        features[f"{prefix}:{hit.gene_symbol}"] = 1
        if hit.drug_class:
            features[f"class:{hit.drug_class}"] = 1
    return features


def featurize_fasta(
    fasta_path: str | Path,
    organism: str | None = None,
    threads: int = 4,
) -> tuple[str, dict[str, int], GenomeQC, list[AMRHit]]:
    """Run the full read -> annotate -> featurize path for one genome.

    Returns the genome id, its binary features, QC record, and the raw hits (kept so
    the app can cite the specific determinant behind each prediction).
    """
    genome_id, stats = read_fasta_stats(fasta_path)
    report = amrfinder.run_amrfinder(fasta_path, organism=organism, threads=threads)
    hits = amrfinder.parse_report(report)
    features = hits_to_features(hits)

    qc = GenomeQC(
        genome_id=genome_id,
        n_amr_genes=sum(1 for h in hits if not h.is_point_mutation),
        **stats,
    )
    return genome_id, features, qc, hits


def build_feature_matrix(
    fasta_dir: str | Path | None = None,
    precomputed_tsv: str | Path | None = None,
    organism: str | None = None,
    threads: int = 4,
    min_prevalence: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the (genomes x features) matrix for a whole cohort.

    Supply either a directory of FASTA files or a precomputed AMRFinderPlus TSV (the
    challenge dataset ships one; using it skips hours of annotation).

    `min_prevalence` drops determinants seen in fewer than N genomes. Singleton genes
    cannot generalize — they only let the model memorize individual isolates.

    Returns (feature_matrix, qc_table).
    """
    if (fasta_dir is None) == (precomputed_tsv is None):
        raise ValueError("Pass exactly one of fasta_dir or precomputed_tsv")

    if precomputed_tsv is not None:
        rows, qc_records = _features_from_precomputed(precomputed_tsv)
    else:
        rows, qc_records = _features_from_fasta_dir(fasta_dir, organism, threads)

    matrix = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    matrix.index.name = "genome_id"
    matrix = matrix.sort_index(axis=1)

    if min_prevalence > 1 and not matrix.empty:
        keep = matrix.columns[matrix.sum(axis=0) >= min_prevalence]
        matrix = matrix[keep]

    qc = pd.DataFrame([asdict(q) for q in qc_records]).set_index("genome_id")
    qc["qc_flags"] = [", ".join(q.flags()) for q in qc_records]
    return matrix, qc


def _features_from_fasta_dir(
    fasta_dir: str | Path, organism: str | None, threads: int
) -> tuple[dict[str, dict[str, int]], list[GenomeQC]]:
    fasta_dir = Path(fasta_dir)
    paths = sorted(
        p for ext in ("*.fasta", "*.fna", "*.fa") for p in fasta_dir.glob(ext)
    )
    if not paths:
        raise FileNotFoundError(f"No FASTA files in {fasta_dir}")

    rows: dict[str, dict[str, int]] = {}
    qc_records: list[GenomeQC] = []
    for i, path in enumerate(paths, 1):
        try:
            genome_id, features, qc, _ = featurize_fasta(path, organism, threads)
            rows[genome_id] = features
            qc_records.append(qc)
        except Exception as exc:  # one bad assembly must not kill the cohort
            print(f"[{i}/{len(paths)}] SKIPPED {path.name}: {exc}")
            continue
        if i % 25 == 0:
            print(f"[{i}/{len(paths)}] annotated")
    return rows, qc_records


def _features_from_precomputed(
    tsv_path: str | Path,
) -> tuple[dict[str, dict[str, int]], list[GenomeQC]]:
    """Parse a pooled AMRFinderPlus TSV that carries a genome_id column."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]

    id_col = next(
        (c for c in ("genome_id", "Genome ID", "Name", "genome") if c in df.columns), None
    )
    if id_col is None:
        raise ValueError(
            f"No genome id column in {tsv_path}. Expected one of: genome_id, Genome ID, Name"
        )

    rows: dict[str, dict[str, int]] = {}
    qc_records: list[GenomeQC] = []
    for genome_id, group in df.groupby(id_col):
        hits = amrfinder.parse_report(amrfinder.normalize_report(group))
        rows[str(genome_id)] = hits_to_features(hits)
        qc_records.append(
            GenomeQC(
                genome_id=str(genome_id),
                n_contigs=-1,
                total_length=-1,
                longest_contig=-1,
                gc_content=-1.0,
                n_amr_genes=sum(1 for h in hits if not h.is_point_mutation),
            )
        )
    return rows, qc_records


def save_features(matrix: pd.DataFrame, qc: pd.DataFrame, out_dir: str | Path) -> None:
    """Persist the feature matrix, QC table, and a schema manifest for reproducibility."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix.to_parquet(out_dir / "features.parquet")
    qc.to_csv(out_dir / "genome_qc.csv")

    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "n_genomes": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "feature_names": list(matrix.columns),
        "feature_families": {
            "gene": int(sum(c.startswith("gene:") for c in matrix.columns)),
            "point": int(sum(c.startswith("point:") for c in matrix.columns)),
            "class": int(sum(c.startswith("class:") for c in matrix.columns)),
        },
    }
    (out_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {matrix.shape[0]} genomes x {matrix.shape[1]} features -> {out_dir}")
