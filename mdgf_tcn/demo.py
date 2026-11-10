"""Run an offline C2-to-C3 forward pass using synthetic candidate regions."""

import numpy as np
import pandas as pd
import torch

from .data import depth_neighborhoods
from .models import C2DirectionalResNet18Top96Fusion, C3DepthContextTCN


def run():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    n = 8
    rows = pd.DataFrame({
        "well_code": ["synthetic"] * n,
        "image_id": ["synthetic-window"] * n,
        "region_center_depth": np.linspace(0.0, 0.3, n),
    })
    sequence, extra = depth_neighborhoods(rows, {"demo": np.arange(n)})
    patches = torch.rand(n, 3, 160, 160)
    means = patches.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    stds = patches.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    attributes = torch.randn(n, 96)
    c2 = C2DirectionalResNet18Top96Fusion().eval()
    c3 = C3DepthContextTCN().eval()
    with torch.inference_mode():
        _, encoded = c2((patches - means) / stds, attributes, return_embedding=True)
        model_input = torch.cat([encoded[torch.from_numpy(sequence)], torch.from_numpy(extra)], dim=-1)
        probabilities = c3(model_input).sigmoid()
    assert encoded.shape == (n, 768) and model_input.shape == (n, 15, 771)
    assert probabilities.shape == (n,) and torch.isfinite(probabilities).all()
    print(f"C2 encodings {tuple(encoded.shape)}")
    print(f"C3 inputs {tuple(model_input.shape)}")
    print("Synthetic probabilities:", probabilities.tolist())
    return probabilities


if __name__ == "__main__":
    run()
