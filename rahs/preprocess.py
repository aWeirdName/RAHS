"""Epoch and whiten THINGS-EEG raw recordings using training-session covariance.

Adapted from UBP's process_eeg_whiten.py through the experimental RAHS source.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import CHANNELS, image_id


def image_manifest(root):
    paths = sorted(p for p in Path(root).glob("*/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise ValueError(f"No category images found under {root}")
    categories = {name: i for i, name in enumerate(sorted({p.parent.name for p in paths}))}
    return [image_id(p) for p in paths], np.array([categories[p.parent.name] for p in paths])


def epoch_session(path, mode, seed):
    import mne
    from sklearn.utils import shuffle

    record = np.load(path, allow_pickle=True).item()
    info = mne.create_info(record["ch_names"], record["sfreq"], record["ch_types"])
    raw = mne.io.RawArray(record["raw_eeg_data"], info, verbose=False)
    events = mne.find_events(raw, stim_channel="stim", verbose=False)
    events = events[events[:, 2] != 99999]
    raw.pick_channels(CHANNELS, ordered=True)
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=1.0, baseline=(None, 0),
                        preload=True, verbose=False)
    epochs.resample(250)
    values = epochs.get_data()[..., -250:]
    event_ids = epochs.events[:, 2]
    conditions = np.unique(event_ids)
    repetitions = 20 if mode == "test" else 2
    grouped = []
    for condition in conditions:
        indices = np.flatnonzero(event_ids == condition)
        if len(indices) < repetitions:
            raise ValueError(f"Condition {condition} has fewer than {repetitions} trials in {path}")
        indices = shuffle(indices, random_state=seed, n_samples=repetitions)
        grouped.append(values[indices])
    return np.stack(grouped), conditions, epochs.times[-250:]


def training_whitener(training):
    from sklearn.covariance import ledoit_wolf
    from scipy.linalg import fractional_matrix_power

    # Equivalent to sklearn.discriminant_analysis._cov(..., shrinkage='auto'),
    # using public APIs: standardize features, estimate LW covariance, rescale.
    from sklearn.preprocessing import StandardScaler

    total = np.zeros((training.shape[2], training.shape[2]))
    for condition in training:
        covariance = []
        for repetition in condition:
            scaler = StandardScaler()
            standardized = scaler.fit_transform(repetition.T)
            cov = ledoit_wolf(standardized)[0]
            covariance.append(scaler.scale_[:, None] * cov * scaler.scale_[None, :])
        total += np.mean(covariance, axis=0)
    covariance = total / len(training)
    whitening = fractional_matrix_power(covariance, -0.5)
    if np.iscomplexobj(whitening):
        if np.max(np.abs(whitening.imag)) > 1e-6:
            raise ValueError("Whitening covariance produced a complex matrix")
        whitening = whitening.real
    if not np.isfinite(whitening).all():
        raise ValueError("Whitening matrix contains non-finite values")
    return whitening


def merge_conditions(session_values, session_conditions, image_count, expected_repetitions):
    grouped = [[] for _ in range(image_count)]
    for values, conditions in zip(session_values, session_conditions):
        for row, condition in zip(values, conditions):
            index = int(condition) - 1
            if not 0 <= index < image_count:
                raise ValueError(f"Condition {condition} is outside the image manifest")
            grouped[index].append(row)
    merged = []
    for index, rows in enumerate(grouped):
        if not rows:
            raise ValueError(f"Image condition {index + 1} has no EEG trials")
        row = np.concatenate(rows, axis=0)
        if len(row) != expected_repetitions:
            raise ValueError(f"Image condition {index + 1}: expected {expected_repetitions} repetitions, got {len(row)}")
        merged.append(row)
    return np.stack(merged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Directory containing sub-XX/ses-XX/raw_eeg_*.npy")
    parser.add_argument("--image-dir", required=True, help="Directory containing training_images and test_images")
    parser.add_argument("--output", required=True, help="Preprocessed_data_250Hz_whiten directory")
    parser.add_argument("--subjects", nargs="+", default=[f"sub-{i:02}" for i in range(1, 11)])
    parser.add_argument("--seed", type=int, default=20200220)
    args = parser.parse_args()
    manifests = {split: image_manifest(Path(args.image_dir) / folder)
                 for split, folder in (("train", "training_images"), ("test", "test_images"))}
    for subject in args.subjects:
        output = Path(args.output) / subject
        if any((output / f"{split}.pt").exists() for split in ("train", "test")):
            raise FileExistsError(f"Preprocessed files already exist in {output}")
        values, conditions = {"train": [], "test": []}, {"train": [], "test": []}
        for session in range(1, 5):
            root = Path(args.raw_dir) / subject / f"ses-{session:02}"
            train, train_ids, times = epoch_session(root / "raw_eeg_training.npy", "train", args.seed)
            test, test_ids, test_times = epoch_session(root / "raw_eeg_test.npy", "test", args.seed)
            if not np.allclose(times, test_times):
                raise ValueError("Train/test sample times differ")
            whitening = training_whitener(train)
            for split, data, ids in (("train", train, train_ids), ("test", test, test_ids)):
                values[split].append((data.swapaxes(-1, -2) @ whitening).swapaxes(-1, -2))
                conditions[split].append(ids)
            print(f"{subject}: session {session}/4", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        for split, repetitions in (("train", 4), ("test", 80)):
            paths, labels = manifests[split]
            eeg = merge_conditions(values[split], conditions[split], len(paths), repetitions)
            torch.save({"eeg": torch.from_numpy(eeg.astype(np.float32)),
                        "img": paths, "label": torch.as_tensor(labels),
                        "ch_names": CHANNELS, "times": torch.as_tensor(times),
                        "metadata": {"seed": args.seed, "whitening": "train-session covariance only",
                                     "sampling_rate": 250, "time_selection": "last 250 samples of [-0.2,1.0] epoch"}},
                       output / f"{split}.pt")


if __name__ == "__main__":
    main()
