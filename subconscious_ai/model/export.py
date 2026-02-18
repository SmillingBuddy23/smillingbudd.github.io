from __future__ import annotations

import argparse

import torch

from features.aggregator import FEATURE_NAMES
from model.net import IntentModel


def export_onnx(weights_path: str, out_path: str) -> None:
    model = IntentModel(input_size=len(FEATURE_NAMES))
    model.load(weights_path)
    dummy = torch.randn(1, 16, len(FEATURE_NAMES), dtype=torch.float32)
    torch.onnx.export(
        model.net,
        dummy,
        out_path,
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch", 1: "time"}},
        opset_version=13,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="artifacts/demo_model.pt")
    p.add_argument("--out", default="artifacts/demo_model.onnx")
    args = p.parse_args()
    export_onnx(args.weights, args.out)
