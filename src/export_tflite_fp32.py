"""
export_tflite_fp32.py

Export CoreSet ST-GCN to FP32 TFLite for stable mobile deployment.
This follows the same logic as the one-off script that worked on mobile.
"""

import argparse
from pathlib import Path

import torch
import litert_torch

from src.models.stgcn_multitask import CoreSetSTGCN_MultiTask


def export_fp32_tflite(checkpoint_path: str, output_path: str) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = CoreSetSTGCN_MultiTask(
        in_channels=14,
        num_classes=4,
        max_frames=64,
        node_count=33,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    sample_input = torch.randn(1, 14, 64, 33, 1)

    with torch.no_grad():
        logits, density = model(sample_input)
        print("PyTorch logits:", logits)
        print("PyTorch density min/max:", density.min().item(), density.max().item())

    edge_model = litert_torch.convert(model, (sample_input,))
    edge_model.export(str(output_path))

    print("Saved:", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export CoreSet ST-GCN to FP32 TFLite"
    )

    parser.add_argument(
        "--checkpoint",
        default="/Users/gelika/VsCode/CoreSet-AI-Fitness-Tracker-ML-Pipeline/checkpoint/stgcn_v005/final_stgcn_model.pth",
        help="Path to the trained ST-GCN checkpoint",
    )

    parser.add_argument(
        "--output",
        default="/Users/gelika/VsCode/CoreSet-AI-Fitness-Tracker-ML-Pipeline/checkpoint/stgcn_v005/stgcn_fp32.tflite",
        help="Output path for the FP32 TFLite model",
    )

    args = parser.parse_args()

    export_fp32_tflite(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )