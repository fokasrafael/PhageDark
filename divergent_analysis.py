#!/usr/bin/env python3
"""
PhageDark — leakage-controlled comparison of ESM-2-650M and Phold
on strictly divergent phage proteins (Sections 2.7.4 and 3.9).

Reproduces Table 6 and the divergent-set statistics reported in the manuscript.

Inputs (paths relative to BASE):
  phrogs/splits/{train,val,test}.tsv        PHROG-cluster-disjoint splits
  embeddings/esm2_650m_embeddings_test.h5   precomputed ESM-2-650M embeddings
  best_classifier_v2.pt                     trained MLP classifier
  blast_test_vs_train_full.tsv              BLASTp test-vs-train (outfmt 6)
  phold_full_backup/compare{0,1,2}/phold_per_cds_predictions.tsv
  phold_full_backup/mapping_full.json       query-id -> protein_id mapping
  empathi_meta/protein_info_dedup.csv       Empathi protein -> PHROG table
  empathi_meta/train_test_splits.xlsx       Empathi train/test membership

Outputs:
  divergent_set.tsv, divergent_final.tsv, divergent_metrics.json

Upstream commands (run before this script):
  blastp -query test.fasta -db traindb -outfmt "6 qseqid sseqid pident length evalue bitscore" \
         -max_target_seqs 1 -evalue 1e-3 -num_threads 8 -out blast_test_vs_train_full.tsv
  phold proteins-predict -i chunk{N}.fasta -o predict{N} -d phold_db -t 8
  phold proteins-compare -i chunk{N}.fasta --predictions_dir predict{N} -o compare{N} -d phold_db -t 8

Empathi was not re-run on this subset: the released version pins torch==2.3.0,
which is unavailable for the Python version of the compute environment used here.
"""

import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

SEED = 42

# Collapse the 11 PhageDark classes onto the 9 native PHROGs categories
OUR11_TO_PHROGS9 = {
    "structural_head_capsid": "head and packaging",
    "packaging_terminase": "head and packaging",
    "structural_tail_fiber_baseplate": "tail",
    "structural_other": "connector",
    "dna_replication_recombination": "DNA, RNA and nucleotide metabolism",
    "lysis": "lysis",
    "integration_excision": "integration and excision",
    "transcription_regulation": "transcription regulation",
    "auxiliary_metabolic_gene": "moron, auxiliary metabolic gene and host takeover",
    "host_interaction_modulation": "moron, auxiliary metabolic gene and host takeover",
    "other_known_function": "other",
}

# Divergence thresholds (Section 2.7.4)
OUR_PIDENT_MAX = 90.0    # BLASTp % identity to our supervised training split
PHOLD_FIDENT_MAX = 0.90  # Foldseek fractional identity to the Phold structure DB


class MLP(nn.Module):
    """Classifier head as trained (config stored in the checkpoint)."""

    def __init__(self, in_dim=1280, n_classes=11, hidden=(1024, 512, 256), dropout=0.5):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def wilson_ci(successes, n, z=1.959963985):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def normalise_phrog(value):
    s = str(value)
    return s if s.startswith("phrog_") else f"phrog_{s}"


def load_phold(base):
    """Concatenate the chunked Phold outputs and attach ground truth."""
    parts = [
        pd.read_csv(base / "phold_full_backup" / f"compare{c}" / "phold_per_cds_predictions.tsv", sep="\t")
        for c in range(3)
    ]
    ph = pd.concat(parts, ignore_index=True)

    mapping = json.loads((base / "phold_full_backup" / "mapping_full.json").read_text())
    ph["protein_id"] = ph["cds_id"].astype(str).map(lambda s: mapping.get(s, {}).get("protein_id"))
    ph["true_category"] = ph["cds_id"].astype(str).map(lambda s: mapping.get(s, {}).get("true_category"))
    ph["phrog"] = ph["cds_id"].astype(str).map(lambda s: normalise_phrog(mapping.get(s, {}).get("phrog_id")))
    return ph


def empathi_training_phrogs(base):
    """PHROG clusters represented in the Empathi training set (Zenodo 10.5281/zenodo.14036011)."""
    info = pd.read_csv(base / "empathi_meta" / "protein_info_dedup.csv")
    id2phrog = dict(zip(info["proteinID"].astype(str), info["#phrog"].astype(str)))

    splits_path = base / "empathi_meta" / "train_test_splits.xlsx"
    book = pd.ExcelFile(splits_path)
    train_ids = set()
    for sheet in book.sheet_names:
        df = pd.read_excel(book, sheet_name=sheet)
        train_ids |= set(df.loc[df["dataset"] == "train", df.columns[0]].astype(str))

    return {id2phrog[p] for p in train_ids if p in id2phrog}


def apply_divergence_filters(ph, base):
    """Exclude proteins that are near-identical to any method's reference data."""
    # (a) our supervised training split
    cols = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"]
    blast = pd.read_csv(base / "blast_test_vs_train_full.tsv", sep="\t", names=cols)
    blast["query"] = blast["qseqid"].str.split().str[0]
    best = blast.sort_values("bitscore", ascending=False).groupby("query").first()
    ph["query"] = ph["protein_id"].astype(str).str.split().str[0]
    ph["our_pident"] = ph["query"].map(best["pident"]).fillna(0.0)
    ph["near_our_train"] = ph["our_pident"] >= OUR_PIDENT_MAX

    # (b) the Phold structure database
    ph["near_phold_db"] = ph["fident"].fillna(0.0) >= PHOLD_FIDENT_MAX

    # (c) the Empathi training set, at PHROG-cluster level
    ph["in_empathi_train"] = ph["phrog"].isin(empathi_training_phrogs(base))

    ph["divergent"] = ~ph["near_our_train"] & ~ph["near_phold_db"] & ~ph["in_empathi_train"]
    return ph


def esm2_predictions(base, protein_ids, device):
    """Run the trained classifier over precomputed ESM-2-650M embeddings."""
    ckpt = torch.load(base / "best_classifier_v2.pt", map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    cfg = ckpt["config"]

    model = MLP(
        n_classes=len(class_to_idx),
        hidden=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    embeddings, kept = [], []
    with h5py.File(base / "embeddings" / "esm2_650m_embeddings_test.h5", "r") as handle:
        for pid in protein_ids:
            if pid in handle:
                embeddings.append(handle[pid][:])
                kept.append(pid)

    preds, confs = [], []
    with torch.no_grad():
        matrix = np.asarray(embeddings, dtype=np.float32)
        for start in range(0, len(matrix), 4096):
            probs = torch.softmax(model(torch.from_numpy(matrix[start:start + 4096]).to(device)), dim=1)
            probs = probs.cpu().numpy()
            preds.append(probs.argmax(1))
            confs.append(probs.max(1))

    preds = np.concatenate(preds) if preds else np.array([])
    confs = np.concatenate(confs) if confs else np.array([])
    return (
        {pid: idx_to_class[i] for pid, i in zip(kept, preds)},
        {pid: float(c) for pid, c in zip(kept, confs)},
    )


def report(label, y_true, y_pred, annotated_mask, n_total):
    """Coverage and accuracy among annotated proteins, with a Wilson interval."""
    n_annotated = int(annotated_mask.sum())
    coverage = n_annotated / n_total if n_total else float("nan")
    if n_annotated == 0:
        return {"method": label, "coverage": coverage, "n_annotated": 0, "accuracy": None}
    acc = accuracy_score(y_true[annotated_mask], y_pred[annotated_mask])
    lo, hi = wilson_ci(round(acc * n_annotated), n_annotated)
    print(f"  {label:<26} coverage={coverage:6.1%} ({n_annotated}/{n_total})  "
          f"accuracy={acc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    return {"method": label, "coverage": coverage, "n_annotated": n_annotated,
            "accuracy": acc, "ci_low": lo, "ci_high": hi}


def main(base):
    base = Path(base)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ph = load_phold(base)
    ph = apply_divergence_filters(ph, base)

    print(f"Test proteins: {len(ph):,}")
    for col, name in [("near_our_train", "near-identical to our training split"),
                      ("near_phold_db", "near-identical in Phold structure DB"),
                      ("in_empathi_train", "in an Empathi training PHROG cluster")]:
        print(f"  excluded, {name}: {ph[col].sum():,} ({ph[col].mean():.1%})")

    div = ph[ph["divergent"]].copy()
    print(f"\nDivergent set: {len(div):,} ({len(div) / len(ph):.1%} of the test set)")

    # Ground truth and predictions in the 9 native PHROGs categories
    div["true9"] = div["true_category"].map(OUR11_TO_PHROGS9)
    div["phold9"] = div["function"].astype(str).str.strip()

    esm_pred, esm_conf = esm2_predictions(base, div["protein_id"].tolist(), device)
    div["esm9"] = div["protein_id"].map(lambda p: OUR11_TO_PHROGS9.get(esm_pred.get(p)))
    div["esm_conf"] = div["protein_id"].map(esm_conf)

    div["esm_annotated"] = div["esm9"].notna()
    div["phold_annotated"] = div["phold9"] != "unknown function"

    results = {"n_divergent": int(len(div))}

    print("\nAll divergent proteins, nine PHROGs categories:")
    results["all"] = [
        report("ESM-2-650M", div["true9"].values, div["esm9"].values, div["esm_annotated"].values, len(div)),
        report("Phold", div["true9"].values, div["phold9"].values, div["phold_annotated"].values, len(div)),
    ]

    # Majority-class reference: the divergent set is dominated by "other"
    majority = div["true9"].value_counts(normalize=True).iloc[0]
    majority_label = div["true9"].value_counts().index[0]
    print(f"  {'majority-class baseline':<26} accuracy={majority:.3f} (always \"{majority_label}\")")
    results["majority_baseline"] = {"label": majority_label, "accuracy": float(majority)}

    # Proteins annotated by both methods
    both = div[div["esm_annotated"] & div["phold_annotated"]]
    acc_esm = accuracy_score(both["true9"], both["esm9"])
    acc_phold = accuracy_score(both["true9"], both["phold9"])
    rng = np.random.default_rng(SEED)
    yt, ye, yp = both["true9"].values, both["esm9"].values, both["phold9"].values
    deltas = np.array([
        accuracy_score(yt[i], ye[i]) - accuracy_score(yt[i], yp[i])
        for i in (rng.integers(0, len(both), len(both)) for _ in range(1000))
    ])
    print(f"\nProteins annotated by both methods (n={len(both)}):")
    print(f"  ESM-2-650M accuracy={acc_esm:.3f}   Phold accuracy={acc_phold:.3f}")
    print(f"  paired bootstrap delta 95% CI [{np.percentile(deltas, 2.5):+.3f}, "
          f"{np.percentile(deltas, 97.5):+.3f}], p={(deltas <= 0).mean():.4f}")
    results["intersection"] = {
        "n": int(len(both)), "esm_accuracy": float(acc_esm), "phold_accuracy": float(acc_phold),
        "delta_ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_value": float((deltas <= 0).mean()),
    }

    # Excluding the dominant category
    spec = div[div["true9"] != majority_label]
    print(f"\nExcluding the dominant \"{majority_label}\" category (n={len(spec)}):")
    results["excluding_majority"] = [
        report("ESM-2-650M", spec["true9"].values, spec["esm9"].values, spec["esm_annotated"].values, len(spec)),
        report("Phold", spec["true9"].values, spec["phold9"].values, spec["phold_annotated"].values, len(spec)),
    ]

    ph.to_csv(base / "phold_full_with_filters.tsv", sep="\t", index=False)
    div[["protein_id", "phrog", "true_category", "true9"]].to_csv(base / "divergent_set.tsv", sep="\t", index=False)
    div.to_csv(base / "divergent_final.tsv", sep="\t", index=False)
    (base / "divergent_metrics.json").write_text(json.dumps(results, indent=2))
    print("\nWrote divergent_set.tsv, divergent_final.tsv, divergent_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=".", help="directory holding the PhageDark data files")
    main(parser.parse_args().base)
