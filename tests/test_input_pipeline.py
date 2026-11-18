"""Checks for the manuscript-size input and split-isolated model workflow."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from mdgf_tcn.data import depth_neighborhoods, load_dataset
from mdgf_tcn.models import C2DirectionalResNet18Top96Fusion, C3DepthContextTCN


class InputPipelineTests(unittest.TestCase):
    def write_dataset(self, folder, size):
        n = 6
        pd.DataFrame({
            "well_code": ["synthetic"] * n,
            "image_id": np.repeat(["w1", "w2", "w3"], 2),
            "source_group_id": np.repeat(["g1", "g2", "g3"], 2),
            "region_center_depth": np.arange(n) * 0.01,
            "label_id": np.tile([0, 1], 3),
        }).to_csv(folder / "regions.csv", index=False)
        np.save(folder / "patches.npy", np.zeros((n, size, size, 3), dtype=np.uint8))
        np.save(folder / "attributes.npy", np.zeros((n, 96), dtype=np.float32))
        np.savez(folder / "splits.npz", train_indices=np.array([0, 1]),
                 val_indices=np.array([2, 3]), test_indices=np.array([4, 5]))

    def test_training_loader_accepts_360_patches(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write_dataset(folder, 360)
            dataset = load_dataset(folder, group_disjoint=True)
            try:
                self.assertEqual(dataset.patches.shape, (6, 360, 360, 3))
            finally:
                dataset.patches._mmap.close()

    def test_training_loader_rejects_old_patch_size(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write_dataset(folder, 160)
            with self.assertRaisesRegex(ValueError, r"N,360,360,3"):
                load_dataset(folder, group_disjoint=False)

    def test_neighborhoods_do_not_cross_subsets(self):
        rows = pd.DataFrame({
            "well_code": ["synthetic"] * 6,
            "image_id": ["shared-window"] * 6,
            "region_center_depth": np.arange(6) * 0.01,
        })
        splits = {"train": np.array([0, 3]), "val": np.array([1, 4]),
                  "test": np.array([2, 5])}
        sequence, auxiliary = depth_neighborhoods(rows, splits)
        for indices in splits.values():
            self.assertTrue(np.isin(sequence[indices], indices).all())
            np.testing.assert_array_equal(sequence[indices, 7], indices)
            np.testing.assert_array_equal(auxiliary[indices, 7, 2], 1)

    def test_360_feature_maps_and_c2_c3_forward(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        c2 = C2DirectionalResNet18Top96Fusion().eval()
        c3 = C3DepthContextTCN().eval()
        observed = []
        hooks = [getattr(c2.backbone, f"layer{i}").register_forward_hook(
            lambda module, inputs, output: observed.append(tuple(output.shape[1:])))
            for i in range(1, 5)]
        try:
            with torch.inference_mode():
                logits, embedding = c2(torch.zeros(2, 3, 360, 360),
                                       torch.zeros(2, 96), return_embedding=True)
                sequence = torch.cat((embedding[:, None, :].expand(-1, 15, -1),
                                      torch.zeros(2, 15, 3)), dim=-1)
                probabilities = c3(sequence).sigmoid()
            self.assertEqual(observed, [(64, 90, 90), (128, 45, 45),
                                        (256, 23, 23), (512, 12, 12)])
            self.assertEqual(tuple(logits.shape), (2,))
            self.assertEqual(tuple(embedding.shape), (2, 768))
            self.assertEqual(tuple(sequence.shape), (2, 15, 771))
            self.assertEqual(tuple(probabilities.shape), (2,))
            self.assertTrue(torch.isfinite(probabilities).all().item())
        finally:
            for hook in hooks:
                hook.remove()


if __name__ == "__main__":
    unittest.main()
