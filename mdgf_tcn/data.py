"""Input validation and the split-isolated depth neighborhoods used in the paper."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SUBSETS = ("train", "val", "test")


@dataclass
class Dataset:
    metadata: pd.DataFrame
    labels: np.ndarray
    patches: np.ndarray
    attributes: np.ndarray
    splits: dict[str, np.ndarray]


def load_dataset(folder: Path, group_disjoint: bool) -> Dataset:
    metadata = pd.read_csv(folder / "regions.csv", dtype={"well_code": str, "image_id": str})
    required = ["well_code", "image_id", "region_center_depth", "label_id"]
    if group_disjoint:
        required.append("source_group_id")
    if any(col not in metadata for col in required) or metadata[required].isna().any().any():
        raise ValueError(f"regions.csv must provide nonmissing columns {required}")
    if not np.isfinite(metadata["region_center_depth"].to_numpy(dtype=float)).all():
        raise ValueError("Region depths must be finite")
    if not metadata["label_id"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    labels = metadata["label_id"].to_numpy(dtype=np.int64)
    attributes = np.load(folder / "attributes.npy", allow_pickle=False).astype(np.float32)
    n = len(metadata)
    if attributes.shape != (n, 96) or not np.isfinite(attributes).all():
        raise ValueError("attributes.npy must contain N finite 96-dimensional vectors")
    with np.load(folder / "splits.npz", allow_pickle=False) as archive:
        splits = {name: archive[f"{name}_indices"] for name in SUBSETS}
    for name, indices in splits.items():
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError(f"{name}_indices must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= n) or np.unique(labels[indices]).size != 2:
            raise ValueError(f"{name}_indices must be in range and include both classes")
    if not np.array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(n)):
        raise ValueError("Each region must belong to exactly one subset")
    if group_disjoint:
        membership = np.empty(n, dtype=np.int64)
        for subset, indices in enumerate(splits.values()):
            membership[indices] = subset
        if metadata.assign(subset=membership).groupby("source_group_id")["subset"].nunique().gt(1).any():
            raise ValueError("A source-window group spans more than one subset")
        if metadata.groupby(["well_code", "image_id"])["source_group_id"].nunique().gt(1).any():
            raise ValueError("A source window belongs to more than one group")
    patches = np.load(folder / "patches.npy", mmap_mode="r", allow_pickle=False)
    if patches.shape != (n, 160, 160, 3) or patches.dtype != np.uint8:
        raise ValueError("patches.npy must have shape (N,160,160,3) and dtype uint8")
    return Dataset(metadata, labels, patches, attributes, splits)


def depth_neighborhoods(metadata: pd.DataFrame, splits: dict[str, np.ndarray]):
    """Return region indices and three auxiliary channels for the 15 positions."""
    n = len(metadata)
    sequence = np.empty((n, 15), dtype=np.int64)
    auxiliary = np.empty((n, 15, 3), dtype=np.float32)
    offsets = np.arange(-7, 8)
    depths = metadata["region_center_depth"].to_numpy(dtype=float)
    for ids in splits.values():
        subset = metadata.iloc[ids][["well_code", "image_id", "region_center_depth"]].assign(row=ids)
        for _, window in subset.groupby(["well_code", "image_id"], sort=False):
            rows = window.sort_values(["region_center_depth", "row"], kind="stable")["row"].to_numpy()
            for position, row in enumerate(rows):
                positions = position + offsets
                neighbors = rows[np.clip(positions, 0, len(rows) - 1)]
                sequence[row] = neighbors
                auxiliary[row, :, 0] = np.clip((depths[neighbors] - depths[row]) / 0.2, -1, 1)
                auxiliary[row, :, 1] = offsets / 7.0
                auxiliary[row, :, 2] = (positions >= 0) & (positions < len(rows))
    return sequence, auxiliary
