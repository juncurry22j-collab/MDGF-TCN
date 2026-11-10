# MDGF-TCN

**Current version** — `v1.0.0`.

MDGF-TCN classifies candidate regions in electrical borehole images as lamination or non-lamination. It combines a multiscale directional ResNet18 image encoder, gated fusion of image and region attributes, and a depth-context temporal convolutional network.

The C2 and C3 model definitions are extracted from the study's `lamination_tabm_labelid_randomsplit.ipynb`. The demo notebook is a short, offline example of these definitions. See [the input and experiment guide](docs/experiments.md) for the two training workflows and their data requirements.

## Installation

The reference software environment uses Python 3.10.20, PyTorch 1.13.1, torchvision 0.14.1, and CUDA 11.7 on Windows. The demo also runs on CPU.

**Hardware for future retraining** — A Windows workstation equipped with an NVIDIA GeForce RTX 4070 GPU with 12 GB of video memory. The archived retraining logs record an NVIDIA GeForce GTX 1650 Ti GPU with 4 GB of video memory.

From the repository directory, install the PyTorch and torchvision wheels for your platform. The commands below use the study versions with CPU wheels.

```sh
python -m pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For the study's CUDA setup, use the corresponding CUDA 11.7 wheels. Jupyter Notebook 7.6.0 is optional and is only needed to open the demo notebook.

## Quick start

```sh
python -m mdgf_tcn.demo
```

The demo generates eight synthetic candidate regions, runs C2 and C3, and prints the `(8, 768)` encoder output, the `(8, 15, 771)` sequence input, and eight lamination probabilities. These probabilities come from randomly initialized weights and are for checking the workflow.

## Experiments

- **Matched ablation** uses seeds 42–46 and compares the full model with variants without direction blocks, image–attribute gating, or depth context.
- **Source-window groups** runs the selected MDGF-TCN refit and checks that source-window groups are disjoint between the three subsets.

Both commands select checkpoints by validation average precision (AP), then fix the decision threshold using validation Matthews correlation coefficient (MCC) before calculating test metrics. They write validation selection records, test predictions and metrics to a new local run directory. See the [experiment guide](docs/experiments.md) for commands, input shapes and output formats.

## Reproducibility scope

This release contains the model source code, experiment runners and a synthetic example. The original study inputs and trained weights are held separately. Reproducing the reported numbers requires the study's prepared patches, region attributes, group assignments, split indices and initialization files. The historical region-random training run, full baseline comparison and candidate search for the grouped benchmark are separate experimental records. See [the input and experiment guide](docs/experiments.md) for scoring archived predictions.

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
