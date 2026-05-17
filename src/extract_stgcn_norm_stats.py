"""
extract_stgcn_norm_stats.py

Extract feat_mean and feat_std from a trained ST-GCN checkpoint.

These normalization stats are required by the mobile app before running
ST-GCN inference:

    normalized = (feature - feat_mean[channel]) / feat_std[channel]

Default output:
    checkpoint/stgcn_v005/model/stgcn_norm_stats.json
"""

import argparse
import json
from pathlib import Path

import torch


def extract_norm_stats(checkpoint_path: str, output_path: str | None = None) -> None:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Default output is inside the same checkpoint version folder,
    # under a model/ subfolder.
    if output_path is None:
        output_path = checkpoint_path.parent / "model" / "stgcn_norm_stats.json"
    else:
        output_path = Path(output_path)

    print("=" * 60)
    print("CoreSet ST-GCN Normalization Stats Extraction")
    print("=" * 60)

    print("\nLoading checkpoint:")
    print(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print("\nCheckpoint keys:")
    print(list(ckpt.keys()))

    feat_mean = ckpt.get("feat_mean")
    feat_std = ckpt.get("feat_std")

    if feat_mean is None or feat_std is None:
        print("\nfeat_mean / feat_std not found directly.")
        print("Searching nested checkpoint dictionaries...")

        for key, value in ckpt.items():
            if isinstance(value, dict):
                print(f"Nested dict: {key} -> {list(value.keys())}")

                if feat_mean is None and "feat_mean" in value:
                    feat_mean = value["feat_mean"]

                if feat_std is None and "feat_std" in value:
                    feat_std = value["feat_std"]

    if feat_mean is None or feat_std is None:
        raise SystemExit("ERROR: feat_mean / feat_std not found in checkpoint.")

    feat_mean = feat_mean.cpu().numpy().reshape(-1).tolist()
    feat_std = feat_std.cpu().numpy().reshape(-1).tolist()

    output = {
        "checkpoint": str(checkpoint_path),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print("\nSaved:")
    print(output_path)

    print("\nfeat_mean length:", len(feat_mean))
    print("feat_std length:", len(feat_std))

    print("\nfeat_mean:")
    print(feat_mean)

    print("\nfeat_std:")
    print(feat_std)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract ST-GCN feat_mean and feat_std from checkpoint"
    )

    parser.add_argument(
        "--checkpoint",
        default="/Users/gelika/VsCode/CoreSet-AI-Fitness-Tracker-ML-Pipeline/checkpoint/stgcn_v005/final_stgcn_model.pth",
        help="Path to the trained ST-GCN checkpoint",
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path. If not provided, saves to "
            "checkpoint/<version>/model/stgcn_norm_stats.json"
        ),
    )

    args = parser.parse_args()

    extract_norm_stats(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
    )