"""Load the organizer's challenge dataset into the shapes the pipeline expects.

The brief says the fixed dataset ships as: one species, 3-5 antibiotics, one final
label per genome-antibiotic pair, genomes grouped by genetic similarity, with fixed
training / calibration / test sets and a hidden test set.

Two things follow from that, and both are handled here:

  * **Labels usually arrive long, not wide.** One row per genome-antibiotic pair is the
    natural shape for "one final label per pair". The pipeline wants a genome x drug
    matrix, so we pivot — and refuse to guess if a pair appears twice with conflicting
    labels.

  * **The split is given, not derived.** When the organizer pins train/calibration/test
    groups, re-deriving our own would discard the very thing that makes the reported
    score comparable across teams. `load_split` uses theirs; clustering is only a
    fallback for when no split ships.

Nothing here invents a label. Rows that cannot be resolved to resistant/susceptible are
dropped and counted out loud, because a silently mangled label file produces metrics
that look perfectly healthy and mean nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Phenotype vocabulary. Values are normalised to lowercase and stripped before lookup.
#
# "Intermediate" deliberately maps to NaN rather than to either class. It means the assay
# itself could not call the isolate; folding it into resistant or susceptible would
# manufacture a certainty the laboratory did not have. That is the same principle as the
# model's no-call, applied to the labels.
_RESISTANT = {"resistant", "r", "1", "1.0", "nonsusceptible", "non-susceptible", "fail",
              "likely to fail", "true"}
_SUSCEPTIBLE = {"susceptible", "s", "0", "0.0", "sensitive", "work", "likely to work",
                "false"}
_UNCERTAIN = {"intermediate", "i", "uncertain", "indeterminate", "unknown", "nd", "na",
              "not tested", "", "nan", "none"}

# Column names seen across BV-BRC exports and organizer-flavoured CSVs.
_GENOME_COLS = ("genome_id", "genome", "genome_name", "sample_id", "sample", "isolate",
                "isolate_id", "accession", "assembly", "assembly_accession", "id")
_DRUG_COLS = ("antibiotic", "drug", "antimicrobial", "antibiotic_name", "drug_name",
              "agent")
_LABEL_COLS = ("resistant_phenotype", "phenotype", "label", "susceptibility", "sir",
               "outcome", "resistance", "measurement_phenotype", "final_label")
_GROUP_COLS = ("cluster", "cluster_id", "genetic_group", "group", "group_id", "lineage",
               "clade", "st", "sequence_type", "mlst", "genetic_cluster")
_SPLIT_COLS = ("split", "set", "fold", "partition", "subset", "dataset")

# Evidence fields. The brief is explicit: use laboratory-measured results, not general
# phenotype fields, which may contain model-generated predictions. Training on predicted
# labels means training a model to imitate another model, and every metric stays healthy
# while meaning nothing.
_EVIDENCE_COLS = ("evidence", "laboratory_typing_method", "typing_method",
                  "testing_method", "method")
_COMPUTATIONAL = ("computational", "predicted", "prediction", "in silico", "in-silico",
                  "model")


def _find(columns, candidates) -> str | None:
    """First column whose normalised name matches a candidate."""
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _to_binary(value) -> float:
    """Map one phenotype cell to 1.0 resistant / 0.0 susceptible / NaN uncertain."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().lower()
    if text in _RESISTANT:
        return 1.0
    if text in _SUSCEPTIBLE:
        return 0.0
    if text in _UNCERTAIN:
        return np.nan
    # An unrecognised token is not silently discarded as "uncertain" — that would hide a
    # vocabulary mismatch across the whole file. The caller reports these.
    return np.nan


def _is_long_format(frame: pd.DataFrame) -> bool:
    """Long means one row per genome-antibiotic pair, with the drug named in a column."""
    return _find(frame.columns, _DRUG_COLS) is not None


def load_labels(
    path: str | Path,
    *,
    lab_measured_only: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read a label file into a genome x drug matrix of 1.0 / 0.0 / NaN.

    Accepts either layout:

      long  genome_id, antibiotic, resistant_phenotype   (one row per pair)
      wide  genome_id as index, one column per antibiotic

    Args:
        lab_measured_only: drop rows whose evidence column marks them computational.
    """
    path = Path(path)
    frame = pd.read_csv(path)
    say = print if verbose else (lambda *a, **k: None)

    if not _is_long_format(frame):
        genome_col = _find(frame.columns, _GENOME_COLS) or frame.columns[0]
        wide = frame.set_index(genome_col)
        # Drop any grouping/split columns that ride along with a wide label file.
        drop = [c for c in wide.columns
                if _find([c], _GROUP_COLS + _SPLIT_COLS) is not None]
        wide = wide.drop(columns=drop)
        labels = wide.map(_to_binary)
        labels.index.name = "genome_id"
        say(f"Labels: wide format, {labels.shape[0]} genomes x {labels.shape[1]} drugs")
        _report_coverage(labels, say)
        return labels

    genome_col = _find(frame.columns, _GENOME_COLS)
    drug_col = _find(frame.columns, _DRUG_COLS)
    label_col = _find(frame.columns, _LABEL_COLS)
    if genome_col is None or label_col is None:
        raise ValueError(
            f"{path.name}: found the antibiotic column ({drug_col!r}) but could not "
            f"identify the genome id and phenotype columns. Columns present: "
            f"{list(frame.columns)}"
        )

    say(f"Labels: long format, {len(frame):,} genome-antibiotic rows")

    if lab_measured_only:
        evidence_col = _find(frame.columns, _EVIDENCE_COLS)
        if evidence_col is not None:
            marker = frame[evidence_col].astype(str).str.lower()
            computational = marker.apply(
                lambda v: any(token in v for token in _COMPUTATIONAL)
            )
            if computational.any():
                say(f"  dropped {computational.sum():,} computationally-predicted rows "
                    f"(from {evidence_col!r}) — the brief requires laboratory-measured "
                    f"results")
                frame = frame[~computational]
        else:
            say("  no evidence/method column found — cannot separate laboratory results "
                "from computational predictions. Verify the source before trusting any "
                "metric computed from these labels.")

    resolved = frame[label_col].map(_to_binary)
    unresolved = frame[label_col][resolved.isna() & frame[label_col].notna()]
    if len(unresolved):
        counts = unresolved.astype(str).str.strip().str.lower().value_counts()
        known_uncertain = [v for v in counts.index if v in _UNCERTAIN]
        unknown = counts.drop(index=known_uncertain, errors="ignore")
        if len(known_uncertain):
            n = int(counts[known_uncertain].sum())
            say(f"  {n:,} rows are intermediate/untested -> left unlabelled")
        if len(unknown):
            say(f"  WARNING: {int(unknown.sum()):,} rows carry an unrecognised phenotype "
                f"and were left unlabelled: {dict(unknown.head(8))}")

    tidy = pd.DataFrame({
        "genome_id": frame[genome_col].astype(str),
        "drug": frame[drug_col].astype(str).str.strip().str.lower(),
        "label": resolved.to_numpy(),
    }).dropna(subset=["label"])

    # One final label per genome-antibiotic pair. A pair appearing twice with different
    # results is a real conflict in the source; averaging or last-wins would bury it.
    duplicated = tidy.groupby(["genome_id", "drug"])["label"].nunique()
    conflicts = duplicated[duplicated > 1]
    if len(conflicts):
        say(f"  WARNING: {len(conflicts):,} genome-antibiotic pairs carry conflicting "
            f"labels; dropped rather than guessed. Example: {conflicts.index[0]}")
        conflicting = set(conflicts.index)
        tidy = tidy[~tidy.set_index(["genome_id", "drug"]).index.isin(conflicting)]

    labels = tidy.pivot_table(
        index="genome_id", columns="drug", values="label", aggfunc="first"
    )
    labels.index.name = "genome_id"
    labels.columns.name = None
    say(f"  pivoted to {labels.shape[0]:,} genomes x {labels.shape[1]} drugs")
    _report_coverage(labels, say)
    return labels


def _report_coverage(labels: pd.DataFrame, say) -> None:
    """Print per-drug tested counts and class balance.

    Class balance is not a formality here. A drug that is 95% resistant will produce a
    high accuracy from a model that has learned nothing, which is why the pipeline
    reports balanced accuracy and PR-AUC instead.
    """
    if labels.empty:
        say("  no usable labels")
        return
    rows = []
    for drug in labels.columns:
        column = labels[drug]
        tested = int(column.notna().sum())
        resistant = int((column == 1).sum())
        rows.append({
            "drug": drug,
            "tested": tested,
            "resistant": resistant,
            "susceptible": tested - resistant,
            "resistant_%": round(100 * resistant / tested, 1) if tested else float("nan"),
        })
    say("\n" + pd.DataFrame(rows).to_string(index=False))


def load_features(
    path: str | Path,
    *,
    min_prevalence: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read a feature matrix, or build one from a precomputed AMRFinderPlus report.

    The brief notes organizers may ship precomputed AMRFinderPlus results, so both are
    handled: a genome x determinant matrix, or a long AMRFinderPlus TSV to be encoded.
    """
    path = Path(path)
    say = print if verbose else (lambda *a, **k: None)

    if path.suffix == ".parquet":
        features = pd.read_parquet(path)
    elif path.suffix in {".tsv", ".txt"}:
        return _features_from_amrfinder(path, min_prevalence=min_prevalence, say=say)
    else:
        head = pd.read_csv(path, nrows=5)
        if _find(head.columns, ("element_symbol", "gene_symbol")) is not None:
            return _features_from_amrfinder(path, min_prevalence=min_prevalence, say=say,
                                            sep=",")
        genome_col = _find(head.columns, _GENOME_COLS) or head.columns[0]
        features = pd.read_csv(path, index_col=genome_col)

    features.index = features.index.astype(str)
    features.index.name = "genome_id"
    features = features.select_dtypes(include=["number", "bool"]).astype(float).fillna(0.0)
    say(f"Features: {features.shape[0]:,} genomes x {features.shape[1]:,} determinants")
    return _drop_rare(features, min_prevalence, say)


def _features_from_amrfinder(path: Path, *, min_prevalence: int, say, sep="\t") -> pd.DataFrame:
    """Encode a combined AMRFinderPlus report into gene:/point:/class: binary features."""
    from src.genome_reader import hits_to_features
    from src.utils.amrfinder import normalize_report, parse_report

    raw = pd.read_csv(path, sep=sep, dtype=str)
    genome_col = _find(raw.columns, _GENOME_COLS + ("name", "#name"))
    if genome_col is None:
        raise ValueError(
            f"{path.name}: no genome id column in the AMRFinderPlus report — cannot tell "
            f"which hit belongs to which genome. Columns: {list(raw.columns)}"
        )

    say(f"Features: encoding {len(raw):,} AMRFinderPlus hits from {path.name}")
    rows = {}
    for genome_id, group in raw.groupby(genome_col):
        hits = parse_report(normalize_report(group))
        rows[str(genome_id)] = hits_to_features(hits)

    features = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    features.index.name = "genome_id"
    say(f"  {features.shape[0]:,} genomes x {features.shape[1]:,} determinants")
    return _drop_rare(features, min_prevalence, say)


def _drop_rare(features: pd.DataFrame, min_prevalence: int, say) -> pd.DataFrame:
    """Drop determinants seen in too few genomes.

    A determinant present in one or two isolates cannot generalize; it can only let the
    model memorize those isolates.
    """
    if min_prevalence <= 1 or features.empty:
        return features
    keep = (features > 0).sum() >= min_prevalence
    dropped = int((~keep).sum())
    if dropped:
        say(f"  dropped {dropped:,} determinants seen in fewer than {min_prevalence} "
            f"genomes ({int(keep.sum()):,} kept)")
    return features.loc[:, keep]


def load_groups(path: str | Path, index=None, *, verbose: bool = True) -> pd.Series:
    """Read the organizer's genetic grouping: genome_id -> group id."""
    frame = pd.read_csv(path)
    genome_col = _find(frame.columns, _GENOME_COLS) or frame.columns[0]
    group_col = _find(frame.columns, _GROUP_COLS)
    if group_col is None:
        remaining = [c for c in frame.columns if c != genome_col]
        if not remaining:
            raise ValueError(f"{Path(path).name}: no group column found")
        group_col = remaining[0]

    groups = frame.set_index(frame[genome_col].astype(str))[group_col]
    groups.index.name = "genome_id"
    groups.name = "cluster"
    if index is not None:
        groups = groups.reindex([str(i) for i in index])
    if verbose:
        print(f"Genetic groups: {groups.nunique()} groups over {groups.notna().sum():,} "
              f"genomes (from {group_col!r})")
    return groups


def load_split(path: str | Path, index=None, *, verbose: bool = True):
    """Read an organizer-provided train/calibration/test assignment.

    Returns a `GroupedSplit`, so the rest of the pipeline cannot tell the difference
    between a given split and a derived one.

    Using the organizer's split matters beyond convenience: it is what makes a reported
    score comparable across teams, and the brief pins one precisely so results are not
    each team's own choice of difficulty.
    """
    from src.utils.clustering import GroupedSplit

    frame = pd.read_csv(path)
    genome_col = _find(frame.columns, _GENOME_COLS) or frame.columns[0]
    split_col = _find(frame.columns, _SPLIT_COLS)
    if split_col is None:
        raise ValueError(
            f"{Path(path).name}: no split column found. Columns: {list(frame.columns)}"
        )

    assignment = frame.set_index(frame[genome_col].astype(str))[split_col]
    assignment = assignment.astype(str).str.strip().str.lower()
    if index is not None:
        keep = [str(i) for i in index]
        assignment = assignment.reindex(keep).dropna()

    def pick(*names) -> list[str]:
        return sorted(assignment[assignment.isin(names)].index.tolist())

    train = pick("train", "training", "fit")
    calibration = pick("calibration", "cal", "valid", "validation", "dev")
    test = pick("test", "testing", "holdout", "held-out", "eval", "evaluation")

    unknown = set(assignment.unique()) - {
        "train", "training", "fit", "calibration", "cal", "valid", "validation", "dev",
        "test", "testing", "holdout", "held-out", "eval", "evaluation",
    }
    if unknown and verbose:
        print(f"  ignoring unrecognised split values: {sorted(unknown)}")

    if not train:
        raise ValueError(f"{Path(path).name}: no training rows found in {split_col!r}")

    if not test and verbose:
        print("  no test rows in this file — consistent with a hidden test set. "
              "Evaluation below is on the calibration split only; treat it as "
              "development feedback, not a held-out score.")

    if verbose:
        print(f"Organizer split: {len(train):,} train / {len(calibration):,} calibration "
              f"/ {len(test):,} test")
    return GroupedSplit(train=train, calibration=calibration, test=test)


def align(features: pd.DataFrame, labels: pd.DataFrame, *, verbose: bool = True):
    """Restrict both tables to genomes present in each, preserving order."""
    features = features.copy()
    labels = labels.copy()
    features.index = features.index.astype(str)
    labels.index = labels.index.astype(str)

    shared = features.index.intersection(labels.index)
    if len(shared) == 0:
        raise ValueError(
            "No genome ids in common between features and labels.\n"
            f"  feature ids look like: {list(features.index[:3])}\n"
            f"  label ids look like:   {list(labels.index[:3])}\n"
            "Check for a prefix or suffix difference between the two files."
        )
    if verbose:
        dropped_f = len(features) - len(shared)
        dropped_l = len(labels) - len(shared)
        print(f"Aligned on {len(shared):,} genomes "
              f"(dropped {dropped_f:,} feature-only, {dropped_l:,} label-only)")
    return features.loc[shared], labels.loc[shared]
