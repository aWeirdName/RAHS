"""Evaluate a saved RAHS checkpoint against the complete test candidate bank."""

import argparse
import json
from pathlib import Path

import torch

from .engine import evaluate, make_dataset, make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eeg-dir")
    parser.add_argument("--feature-dir")
    parser.add_argument("--output", required=True, help="New JSON metrics file")
    parser.add_argument("--export-features", help="Optional new .pt feature export")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    for path in (args.output, args.export_features):
        if path and Path(path).exists():
            parser.error(f"Refusing to overwrite {path}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    for key in ("eeg_dir", "feature_dir"):
        if getattr(args, key) is not None:
            config["data"][key] = getattr(args, key)
    dataset = make_dataset(config, "test")
    model = make_model(config, dataset).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    for path in (args.output, args.export_features):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    results = {"protocol": config["evaluation"]["protocol"], "selected_epoch": checkpoint["epoch"],
               "test": evaluate(model, dataset, config, args.device, export_path=args.export_features)}
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
