# Inputs and experiments

This guide applies to MDGF-TCN `v1.0.0`.

## Prepared inputs

Place the following files in a local directory. Keep all files in the same row order.

| File | Contents |
| --- | --- |
| `regions.csv` | One row per candidate region. Required columns are `well_code`, `image_id`, `region_center_depth` (m) and `label_id` (0 for non-lamination, 1 for lamination). The group protocol also requires `source_group_id`. |
| `patches.npy` | Unsigned 8-bit array of shape `(N, 160, 160, 3)`. The channels are grayscale, dark-line response and bright-line response, in that order. Use the study's prepared candidate-region patches. |
| `attributes.npy` | Finite array of shape `(N, 96)`. Use the study's standardized Top-96 attributes for ablation and the training-fitted quantile-transformed Top-96 attributes for the selected group refit. |
| `splits.npz` | Integer vectors `train_indices`, `val_indices`, and `test_indices` using zero-based row positions. Each subset must contain both classes. |

For the group protocol, create `source_group_id` from the study's group assignment. Windows from the same well belong to the same group when they share source rows from the same file and channel or have overlapping depth intervals, including any connections through intermediate windows. Assign entire groups to subsets. Use preprocessing and attribute selection fitted on the training subset for the protocol.

The runner normalizes the three patch channels using the study's fixed mean and standard deviation. It builds 15-position neighborhoods within the same split, well and source window. A sequence position carries a 768-dimensional encoding, a clipped relative depth, a scaled relative position and a validity indicator. At either window edge the nearest available region is repeated and its padding position is marked invalid.

## Matched five-seed ablation

The backbone argument points to the common C0 checkpoint fitted on the original region-random training subset. Only load checkpoints from trusted sources.

```sh
python -m mdgf_tcn.experiment --protocol ablation --dataset private/random --backbone private/c0_resnet18.pt --output runs/ablation
```

The runner trains each seed with the same C2 initialization and corresponding training schedule. It replaces both directional modules with identity operations for the direction ablation. The no-gate variant replaces gated fusion with concatenation and projection, preserving the 768-dimensional output. The center-only variant shares the full model's C2 encoder and uses the center projection and classification head without the temporal or context-gating path. The original study reused the audited matched records for seed 42. This portable runner refits seed 42 alongside the other four seeds.

## Selected source-window-group refit

Use the ImageNet ResNet18 checkpoint corresponding to the initialization of the selected `G1_generic_Q96_C3` refit.

```sh
python -m mdgf_tcn.experiment --protocol group --dataset private/group --backbone private/resnet18-f37072fd.pth --output runs/group
```

This command trains the selected MDGF-TCN configuration on the supplied grouped split. The full paper benchmark also used other candidate configurations and baseline models.

## Outputs

The output directory contains C2 and C3 checkpoints, epoch histories, a `validation_selection.json` record, per-run `test_predictions.npz` files, and a combined `test_metrics.csv`. Ablation also writes `ablation_summary.csv` and `paired_differences.csv`. The prediction files contain row indices, reference labels, continuous scores, the fixed validation threshold and predicted labels. The metric columns include accuracy, precision, recall, F1, MCC, AUROC and AP.

The study files and checkpoints are needed to reproduce the reported benchmark values. The synthetic example can be run without them. Exact retraining outcomes can depend on the PyTorch build, the device and the prepared inputs.

## Scoring archived predictions

If you have an original prediction archive, the evaluation module recalculates the metrics using the stored scores and decision threshold.

```sh
python -m mdgf_tcn.evaluation private/c3r_test_predictions.npz
```

The historical region-random test scores have half-precision quantization, even though the archive stores them as 64-bit values. Full-precision inference can change score ties and hence ranking metrics such as AP. Use the archived scores to reproduce the historical table.
