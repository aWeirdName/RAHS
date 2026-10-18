"""Single-device training and full-bank retrieval evaluation."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from .data import EEGDataset, PooledEEGDataset, load_features, split_training_conditions
from .losses import alignment_loss
from .metrics import retrieval_metrics
from .model import RAHSModel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return config


def make_dataset(config, split, subjects=None):
    data = config["data"]
    subjects = subjects or [data["subject"]]
    cache = load_features(Path(data["feature_dir"]) / f"image_latent_{split}.pt")
    datasets = [EEGDataset(
        Path(data["eeg_dir"]) / subject / f"{split}.pt",
        Path(data["feature_dir"]) / f"image_latent_{split}.pt",
        channels=data["channels"], time_window=data["time_window"],
        average_repetitions=data.get(f"{split}_average", True),
        allow_legacy_index_alignment=data.get("allow_legacy_index_alignment", False),
        feature_cache=cache,
    ) for subject in subjects]
    return datasets[0] if len(datasets) == 1 else PooledEEGDataset(datasets)


def make_model(config, dataset):
    options = dict(config["model"])
    options["channels"] = len(dataset.channels)
    options["time_samples"] = dataset.time_samples
    model = RAHSModel(**options)
    if tuple(dataset.tokens.shape[1:]) != (model.num_tokens, model.dim):
        raise ValueError("Feature cache token count/dimension does not match model configuration")
    return model


def loader(dataset, config, shuffle=False):
    return DataLoader(dataset, batch_size=config["train"]["batch_size"] if shuffle else config["evaluation"]["batch_size"],
                      shuffle=shuffle, num_workers=config["data"].get("num_workers", 0),
                      pin_memory=False, drop_last=False)


@torch.no_grad()
def evaluate(model, dataset, config, device, indices=None, export_path=None):
    model.eval()
    indices = torch.arange(len(dataset)) if indices is None else torch.as_tensor(indices)
    if not len(indices):
        raise ValueError("Evaluation subset is empty")
    # One candidate per image, even when multiple EEG repetitions are evaluated.
    candidate_indices, targets = dataset.feature_index[indices].unique(sorted=True, return_inverse=True)
    candidates = dataset.tokens[candidate_indices].flatten(1)
    queries = []
    for batch in loader(Subset(dataset, indices.tolist()), config):
        global_features, _ = model(batch["eeg"].to(device))
        queries.append(global_features.cpu())
    queries = torch.cat(queries)
    metrics = retrieval_metrics(queries, candidates, targets, config["evaluation"]["batch_size"])
    if export_path:
        torch.save({"eeg_features": queries, "image_features": candidates,
                    "targets": targets, "candidate_feature_indices": candidate_indices}, export_path)
    return metrics


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train(config, output, device):
    seed_everything(config["seed"])
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    setting = config.get("setting", "intra-subject")
    if setting not in {"intra-subject", "inter-subject"}:
        raise ValueError(f"Unknown setting: {setting}")
    subjects = None
    if setting == "inter-subject":
        subjects = [s for s in config["data"]["all_subjects"] if s != config["data"]["subject"]]
        if config["data"]["subject"] not in config["data"]["all_subjects"] or not subjects:
            raise ValueError("Held-out subject must belong to all_subjects, with at least one training subject")
    dataset = make_dataset(config, "train", subjects)
    model = make_model(config, dataset).to(device)
    protocol = config["evaluation"]["protocol"]
    if protocol == "heldout":
        train_indices, val_indices = split_training_conditions(dataset, config["evaluation"]["validation_fraction"], config["seed"])
        val_dataset = dataset
    elif protocol == "legacy_test_selection":
        train_indices = torch.arange(len(dataset))
        val_dataset = make_dataset(config, "test", subjects)
        val_indices = torch.arange(len(val_dataset))
    else:
        raise ValueError(f"Unknown evaluation protocol: {protocol}")
    with (output / "split.json").open("w") as handle:
        json.dump({"protocol": protocol, "train_indices": train_indices.tolist(),
                   "validation_indices": val_indices.tolist()}, handle)
    train_loader = loader(Subset(dataset, train_indices.tolist()), config, shuffle=True)
    options = config["train"]
    if options["epochs"] < 1 or options["batch_size"] < 2 or options["patience"] < 1:
        raise ValueError("epochs and patience must be positive; training batch size must be at least two")
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    best, stale = -1.0, 0
    for epoch in range(options["epochs"]):
        model.train()
        total_loss, count = 0.0, 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            global_features, fine_features = model(batch["eeg"])
            loss, _ = alignment_loss(model, global_features, fine_features,
                                     batch["image_tokens"], batch["importance"],
                                     image_indices=batch["image_index"] if options.get("multi_positive", True) else None,
                                     **config["alignment"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            size = len(batch["eeg"])
            total_loss += loss.item() * size
            count += size
        metrics = evaluate(model, val_dataset, config, device, val_indices)
        record = {"epoch": epoch + 1, "train_loss": total_loss / count, "validation": metrics}
        print(json.dumps(record), flush=True)
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        improved = metrics["top1"] > best + options.get("min_delta", 0.001)
        if improved:
            best, stale = metrics["top1"], 0
        else:
            stale += 1
        checkpoint = {"format_version": 1, "model": model.state_dict(), "config": config,
                      "epoch": epoch + 1, "validation": metrics,
                      "best_validation_top1": best}
        save_checkpoint(output / "last.pt", checkpoint)
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
        if stale >= options["patience"]:
            break
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_dataset = make_dataset(config, "test")
    results = {"protocol": protocol, "selected_epoch": checkpoint["epoch"],
               "test": evaluate(model, test_dataset, config, device)}
    (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    return results
