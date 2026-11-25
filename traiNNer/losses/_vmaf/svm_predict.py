"""Frozen v0.6.1 regressor from Netflix's official model file (see LICENSE.Netflix)."""

import json
from pathlib import Path

import torch
from torch import nn


class SVMPredict(nn.Module):
    def __init__(self):
        super().__init__()
        model = json.loads(Path(__file__).with_name("vmaf_v0.6.1.json").read_bytes())["model_dict"]
        header, support_vectors = model["model"].split("\nSV\n")
        settings = dict(line.split(maxsplit=1) for line in header.splitlines())
        self.gamma = float(settings["gamma"])
        self.rho = float(settings["rho"])
        self.slope, *slopes = model["slopes"]
        self.intercept, *intercepts = model["intercepts"]
        coefficients, vectors = [], []
        for line in support_vectors.splitlines():
            coefficient, *features = line.split()
            row = [0.0] * len(slopes)
            for item in features:
                index, value = item.split(":")
                row[int(index) - 1] = float(value)
            coefficients.append(float(coefficient))
            vectors.append(row)
        for name, data in (("slopes", slopes), ("intercepts", intercepts),
                           ("alpha", coefficients), ("support_vectors", vectors)):
            self.register_buffer(name, torch.tensor(data, dtype=torch.float32))

    def forward(self, x):
        normalized = self.slopes * x + self.intercepts
        distance = (normalized.unsqueeze(1) - self.support_vectors).square().sum(-1)
        score = (self.alpha * (-self.gamma * distance).exp()).sum(-1, keepdim=True)
        return (score - self.rho - self.intercept) / self.slope
