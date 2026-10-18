"""Validated THINGS-EEG tensors and image-token caches."""

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset


CHANNELS = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
    "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
    "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
    "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]
# Names are saved with all new preprocessed files. This fallback is for legacy caches.
POSTERIOR_CHANNELS = CHANNELS[-17:]


def image_id(path):
    """Portable condition ID: category directory plus image filename."""
    return "/".join(str(path).replace("\\", "/").split("/")[-2:])


def load_features(path):
    features = torch.load(path, map_location="cpu", weights_only=True)
    tokens = features["Latents"].float()
    importance = features["token_importance"].float()
    if tokens.ndim != 3 or importance.shape != tokens.shape[:2]:
        raise ValueError("Feature cache must contain Latents [N,T,D] and token_importance [N,T]")
    if not torch.isfinite(tokens).all() or not torch.isfinite(importance).all():
        raise ValueError("Feature cache contains non-finite values")
    if (importance < 0).any() or (importance.sum(1) <= 0).any():
        raise ValueError("Importance must be nonnegative with positive mass per image")
    ids = features.get("image_ids")
    if ids is not None:
        ids = [image_id(x) for x in ids]
        if len(ids) != len(tokens) or len(set(ids)) != len(ids):
            raise ValueError("Feature image_ids must be unique and match the number of images")
    return tokens, importance, ids


class EEGDataset(Dataset):
    """Load one subject/split without duplicating image tokens across repetitions.

    EEG files use the original trusted local .pt format, which may contain NumPy
    arrays. Image features use a tensor-only cache and a portable image manifest.
    """

    def __init__(self, eeg_path, feature_path, channels=POSTERIOR_CHANNELS,
                 time_window=(0, 250), average_repetitions=True,
                 allow_legacy_index_alignment=False, feature_cache=None):
        raw = torch.load(eeg_path, map_location="cpu", weights_only=False)
        eeg = torch.as_tensor(raw["eeg"])
        if eeg.ndim != 4:
            raise ValueError("EEG must have shape [images, repetitions, channels, time]")
        n, repeats, c, t = eeg.shape
        names = list(raw.get("ch_names", CHANNELS))
        if len(names) != c or len(set(names)) != c:
            raise ValueError("Channel metadata does not match the EEG array")
        selected = names if channels is None else list(channels)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("Selected channels must be nonempty and unique")
        missing = set(selected) - set(names)
        if missing:
            raise ValueError(f"Channels missing from EEG file: {sorted(missing)}")
        start, end = time_window
        if not 0 <= start < end <= t:
            raise ValueError(f"Invalid sample window {time_window} for {t} samples")
        self.eeg = eeg[:, :, [names.index(ch) for ch in selected], start:end].float()
        if not torch.isfinite(self.eeg).all():
            raise ValueError("EEG contains non-finite values")
        if average_repetitions:
            self.eeg = self.eeg.mean(1)
        else:
            self.eeg = self.eeg.flatten(0, 1)
        self.tokens, self.importance, feature_ids = load_features(feature_path) if feature_cache is None else feature_cache
        paths = np.asarray(raw["img"])
        if paths.ndim == 2:
            if paths.shape != (n, repeats) or not (paths == paths[:, :1]).all():
                raise ValueError("All repetitions of a condition must refer to the same image")
            paths = paths[:, 0]
        if paths.shape != (n,):
            raise ValueError("Image paths must match EEG conditions")
        condition_ids = [image_id(x) for x in paths]
        if len(set(condition_ids)) != n:
            raise ValueError("EEG conditions must have unique image IDs")
        if feature_ids is None:
            if not allow_legacy_index_alignment:
                raise ValueError("Feature cache lacks image_ids. Regenerate it or explicitly enable legacy index alignment")
            if len(self.tokens) != n:
                raise ValueError("Legacy feature count differs from EEG condition count")
            warnings.warn("Using legacy positional alignment; verify image order independently", stacklevel=2)
            mapping = torch.arange(n)
        else:
            lookup = {key: i for i, key in enumerate(feature_ids)}
            absent = set(condition_ids) - set(lookup)
            if absent:
                raise ValueError(f"EEG image IDs missing from feature cache: {sorted(absent)[:5]}")
            mapping = torch.tensor([lookup[key] for key in condition_ids])
        labels = np.asarray(raw["label"])
        labels = labels[:, 0] if labels.ndim == 2 else labels
        if labels.shape != (n,):
            raise ValueError("Labels must match the number of EEG conditions")
        self.condition_labels = torch.as_tensor(labels.astype(np.int64))
        self.condition_ids = condition_ids
        self.condition_index = torch.arange(n)
        self.feature_index = mapping
        if not average_repetitions:
            self.condition_index = self.condition_index.repeat_interleave(repeats)
            self.feature_index = mapping.repeat_interleave(repeats)
        self.channels = selected
        self.time_samples = end - start

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, index):
        feature = self.feature_index[index]
        return {"eeg": self.eeg[index], "image_tokens": self.tokens[feature],
                "importance": self.importance[feature], "image_index": feature,
                "condition_index": self.condition_index[index]}


def split_training_conditions(dataset, fraction, seed):
    """Hold out whole categories, keeping every repetition together."""
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    labels = dataset.condition_labels
    groups = labels.unique(sorted=True)
    if len(groups) < 2:
        raise ValueError("At least two categories are needed for held-out validation")
    generator = torch.Generator().manual_seed(seed)
    groups = groups[torch.randperm(len(groups), generator=generator)]
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    is_val = torch.isin(labels[dataset.condition_index], groups[:count])
    return (~is_val).nonzero().flatten(), is_val.nonzero().flatten()


class PooledEEGDataset(Dataset):
    """Share a candidate bank across subjects with matching image manifests."""

    def __init__(self, datasets):
        from torch.utils.data import ConcatDataset
        first = datasets[0]
        for dataset in datasets[1:]:
            if (dataset.channels != first.channels or dataset.time_samples != first.time_samples
                    or dataset.condition_ids != first.condition_ids
                    or not torch.equal(dataset.condition_labels, first.condition_labels)):
                raise ValueError("Subjects must share channel, time, and image-condition metadata")
        self.datasets = ConcatDataset(datasets)
        self.tokens, self.importance = first.tokens, first.importance
        self.channels, self.time_samples = first.channels, first.time_samples
        self.condition_ids, self.condition_labels = first.condition_ids, first.condition_labels
        self.condition_index = torch.cat([d.condition_index for d in datasets])
        self.feature_index = torch.cat([d.feature_index for d in datasets])

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]
