# PhageDark

Code, data splits, and trained models for:

> Fokas R., Kalatzis P., Vantarakis A. *Benchmarking protein language models for the functional annotation of bacteriophage proteins under controlled homology leakage.* (submitted)

Protein language model (PLM) embeddings are benchmarked against homology-based
annotation on a PHROG-cluster-disjoint split of PHROGs v4, and compared with the
structure-informed tool Phold and the embedding-based tool Empathi on strictly
divergent proteins.

## Summary of results

| | |
|---|---|
| Source database | PHROGs v4 — 38,880 orthologous groups, 868,340 proteins, 17,473 genomes |
| Held-out test set | 34,288 proteins, PHROG-cluster-disjoint from training |
| ESM-2-650M + MLP head | macro F1 = 0.654, coverage 100% |
| Homology baselines | macro F1 = 0.20–0.41, coverage 29–44% |
| No detectable homolog in training split | macro F1 = 0.592 (ESM-2-650M) |
| Calibration | temperature T = 2.81; ECE 0.214 → 0.049 |

On 834 strictly divergent proteins scored in the nine native PHROGs categories,
ESM-2-650M and Phold are complementary rather than ranked: the classifier
annotates every protein but converges on the dominant "other" category
(accuracy 0.826, against a majority-class baseline of 0.825), whereas Phold
abstains on 75% of inputs and is roughly twice as accurate on proteins carrying
a specific function (0.682 vs 0.336).

## Repository contents

### Data splits

| File | Description |
|---|---|
| `train_ids_labels.tsv` | Training split — protein IDs, PHROG cluster, class label |
| `val.tsv` | Validation split |
| `test.tsv` | Held-out test split (n = 34,288), cluster-disjoint from training |

Protein sequences and ESM-2 embeddings are not included because of size; they
are regenerated from PHROGs v4 using the accessions in these files.

### Models

| File | Description |
|---|---|
| `best_classifier_v2.pt` | Trained MLP head — hidden dims [1024, 512, 256], dropout 0.5, lr 5e-4; includes `class_to_idx` and the training config |
| `calibration_v2.json` | Fitted temperature-scaling parameter and calibration metrics |

### Benchmark results

| File | Description |
|---|---|
| `regenerated_metrics.json` | Primary ESM-2-650M test metrics, regenerated from the released checkpoint |
| `results_blast.json` | BLASTp baseline |
| `diamond_results.json` | DIAMOND baseline, three sensitivity modes |
| `phrogs_mmseqs_FAIR.json` | PHROGs-MMseqs2 profile search, restricted to training-set clusters |
| `cv_5fold_CLUSTER.json` | Five-fold StratifiedGroupKFold cross-validation, PHROG cluster as group |
| `model_size_comparison.json` | ESM-2-150M / 650M / ProtT5-XL comparison |
| `results_lora.json` | LoRA fine-tuning ablation |
| `esm2_subsample_results.json` | ESM-2-650M on the class-stratified comparison subsample (n = 4,999) |
| `phold_results.json` | Phold on the comparison subsample |
| `empathi_results.json` | Empathi on the comparison subsample, across confidence thresholds |
| `tool_comparison_summary.json` | Collected tool-comparison metrics |

### Divergent-protein comparison

| File | Description |
|---|---|
| `divergent_analysis.py` | Reproduces the divergent-set statistics and the tool-comparison table |
| `blast_test_vs_train_full.tsv` | BLASTp of the test split against the training split (outfmt 6) |
| `phold_full_predictions.tsv` | Phold per-CDS predictions over the full test split |
| `phold_full_with_filters.tsv` | The same, annotated with the three divergence criteria |
| `divergent_set.tsv` | The 834 proteins passing all three criteria |
| `divergent_final.tsv` | Divergent set with predictions from both methods in the nine PHROGs categories |

A protein is retained as divergent only if it has no near-identical match in any
method's reference data: BLASTp identity < 90% to the supervised training split,
Foldseek fractional identity < 0.90 to the Phold structure database, and its
PHROG cluster absent from the Empathi training set.

### Predictions on unannotated proteins

| File | Description |
|---|---|
| `unknown_predictions.tsv.gz` | Calibrated predictions for PHROGs proteins of unknown function |
| `test_predictions_esm2_v2.tsv` | Per-protein test-set predictions with confidences |

### Notebook

`phagedark_pipeline.ipynb` — the Colab notebook used for embedding extraction,
training, and the baseline benchmarks.

## Reproducing

```bash
pip install torch numpy pandas h5py scikit-learn
python divergent_analysis.py --base /path/to/data
```

`divergent_analysis.py` documents the upstream BLASTp and Phold commands in its
module docstring. External tools used: BLAST+ v2.12, DIAMOND v2.1.8, MMseqs2,
Phold v1.2.6 with Foldseek v10.941cd33, Empathi v1.0.6.

## Known limitations

Empathi was not re-run on the divergent subset or over the full test set: the
released version pins `torch==2.3.0`, which has no distribution for the Python
version of the compute environment used here. Empathi results are therefore
reported only on the class-stratified subsample. These and the remaining
limitations are stated in the manuscript.

## Data sources

- PHROGs v4 — https://phrogs.lmge.uca.fr
- Phold database — https://github.com/gbouras13/phold
- Empathi training metadata — https://doi.org/10.5281/zenodo.14036011

## License

See `LICENSE`.

## Contact

Rafail Fokas — Laboratory of Public Health, Epidemiology and Quality of Life,
Medical School, University of Patras.
