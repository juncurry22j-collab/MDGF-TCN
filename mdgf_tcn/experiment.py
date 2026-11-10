"""Run the matched ablation or the selected source-group MDGF-TCN configuration."""

import argparse
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
import torch

from .data import depth_neighborhoods, load_dataset
from .evaluation import metrics, write_predictions
from .models import C2DirectionalResNet18Top96Fusion, C3DepthContextTCN
from .training import CenterOnly, DEV, embeddings, put, train_context, train_image


ABLATIONS = {
    "full": "Full MDGF-TCN",
    "no_direction": "w/o Direction",
    "no_gate": "w/o I-A gate",
    "no_context": "w/o depth context",
}


def settings(protocol, seed, backbone, positive_weight):
    c2 = dict(
        name="C2", experiment=protocol, seed=seed, initialization=str(backbone),
        warm_full=False, max_epochs=16, patience=5,
    )
    c3 = dict(name="C3", experiment=protocol, seed=seed, lr=0.0001, max_epochs=32, patience=6)
    if protocol == "ablation":
        c2.update(backbone_lr=2e-5, head_lr=2e-4, wd=0.001, pw=positive_weight, gain_aug=0.0)
        c3.update(dropout=0.20, wd=0.001, pw=positive_weight, noise=0.01)
    else:
        c2.update(backbone_lr=1e-5, head_lr=1.5e-4, wd=0.003, pw=1.0, gain_aug=0.02,
                  image_dropout=0.40, fusion_dropout=0.45)
        c3.update(dropout=0.45, wd=0.003, pw=1.0, noise=0.015)
    return c2, c3


def fit(dataset, protocol, output, backbone, seeds, epoch_limit=None):
    labels, splits, attributes = dataset.labels, dataset.splits, dataset.attributes
    train_labels = labels[splits["train"]]
    positive_weight = float(np.sqrt((train_labels == 0).sum() / (train_labels == 1).sum()))
    seq, aux = depth_neighborhoods(dataset.metadata, splits)
    image_models = {"C2DirectionalResNet18Top96Fusion": C2DirectionalResNet18Top96Fusion}
    variants = ("full", "no_direction", "no_gate") if protocol == "ablation" else ("full",)
    records = []
    for seed in seeds:
        c2, c3 = settings(protocol, seed, backbone, positive_weight)
        if epoch_limit is not None:
            c2["max_epochs"] = min(epoch_limit, c2["max_epochs"])
            c3["max_epochs"] = min(epoch_limit, c3["max_epochs"])
        for variant in variants:
            c2_config = dict(c2, name=f"C2_{variant}", variant=variant)
            c2_dir = output / f"seed_{seed}" / f"C2_{variant}"
            train_image(image_models, dataset.patches, attributes, labels, splits, c2_dir, c2_config)
            path = embeddings(image_models, dataset.patches, attributes, labels, splits, c2_dir, c2_config)
            heads = (variant, "no_context") if variant == "full" and protocol == "ablation" else (variant,)
            for head in heads:
                config = dict(c3, name=f"C3_{head}", embedding_path=str(path), no_context=head == "no_context")
                folder = output / f"seed_{seed}" / config["name"]
                record = train_context(np.load(path), seq, aux, labels, splits, folder, config, C3DepthContextTCN)
                records.append(dict(seed=seed, variant=head, c2_dir=str(c2_dir), c2=c2_config, c3=record))
    put(output / "validation_selection.json", records)

    # Freeze all checkpoints and validation thresholds before generating any test scores.
    test = splits["test"]
    rows = []
    for item in records:
        c2_dir = Path(item["c2_dir"])
        path = embeddings(image_models, dataset.patches, attributes, labels, splits, c2_dir, item["c2"], ids=test)
        encodings = np.load(path)
        record = item["c3"]
        config = record["configuration"]
        model = C3DepthContextTCN(dropout=config["dropout"])
        if config["no_context"]:
            model = CenterOnly(model)
        model.load_state_dict(torch.load(record["path"], map_location="cpu")["model_state_dict"])
        model.to(DEV).eval()
        scores = []
        with torch.inference_mode():
            for ids in np.array_split(test, (len(test) + 127) // 128):
                batch = np.concatenate([encodings[seq[ids]], aux[ids]], axis=-1)
                with torch.cuda.amp.autocast(enabled=DEV.type == "cuda"):
                    logits = model(torch.from_numpy(batch).to(DEV))
                scores.extend(logits.float().sigmoid().cpu().numpy())
        threshold = record["validation"]["threshold"]
        write_predictions(Path(record["path"]).parent / "test_predictions.npz", test, labels, scores, threshold)
        rows.append(dict(Seed=item["seed"], Variant=ABLATIONS[item["variant"]], **metrics(labels[test], scores, threshold)))
    table = pd.DataFrame(rows)
    table.to_csv(output / "test_metrics.csv", index=False)
    if protocol == "ablation" and len(seeds) > 1:
        numeric = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUROC", "AP"]
        table.groupby("Variant", sort=False)[numeric].agg(["mean", "std"]).to_csv(output / "ablation_summary.csv")
        full = table[table["Variant"] == ABLATIONS["full"]].set_index("Seed")
        paired = []
        for name, group in table[table["Variant"] != ABLATIONS["full"]].groupby("Variant", sort=False):
            difference = full[numeric] - group.set_index("Seed")[numeric]
            for metric in numeric:
                values = difference[metric].tolist()
                paired.append(dict(Variant=name, Metric=metric, Mean=mean(values), SD=stdev(values),
                                   FullHigher=sum(value > 0 for value in values)))
        pd.DataFrame(paired).to_csv(output / "paired_differences.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("ablation", "group"), required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="directory of prepared study inputs")
    parser.add_argument("--backbone", type=Path, required=True, help="trusted C0 or ImageNet ResNet18 checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if not args.backbone.is_file():
        parser.error("backbone checkpoint is missing")
    if args.output.exists():
        parser.error("choose an output directory that does not exist")
    dataset = load_dataset(args.dataset, group_disjoint=args.protocol == "group")
    args.output.mkdir(parents=True)
    seeds = (42, 43, 44, 45, 46) if args.protocol == "ablation" else (42,)
    table = fit(dataset, args.protocol, args.output, args.backbone.resolve(), seeds)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
