"""Train RAHS: python -m rahs.train --config configs/rahs.yaml."""

import argparse

import torch

from .engine import load_config, train


def main():
    parser = argparse.ArgumentParser(description="Train RAHS on one THINGS-EEG subject")
    parser.add_argument("--config", default="configs/rahs.yaml")
    parser.add_argument("--subject", help="Subject directory, e.g. sub-01")
    parser.add_argument("--eeg-dir")
    parser.add_argument("--feature-dir")
    parser.add_argument("--output", required=True, help="New/empty experiment directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("subject", "eeg_dir", "feature_dir"):
        value = getattr(args, key)
        if value is not None:
            config["data"][key] = value
    if args.seed is not None:
        config["seed"] = args.seed
    train(config, args.output, torch.device(args.device))


if __name__ == "__main__":
    main()
