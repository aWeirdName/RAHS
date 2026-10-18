# Data contract

```text
data/things-eeg/
  Raw_data/sub-01/ses-01/raw_eeg_training.npy
  Raw_data/sub-01/ses-01/raw_eeg_test.npy
  Image_set/training_images/<category>/<image>.jpg
  Image_set/test_images/<category>/<image>.jpg
  Preprocessed_data_250Hz_whiten/sub-01/train.pt
  Preprocessed_data_250Hz_whiten/sub-01/test.pt
  image_features/image_latent_train.pt
  image_features/image_latent_test.pt
```

All paths are configurable. EEG preprocessing requires four sessions per participant. Standard data contain 16,540 training images with 4 repetitions each and 200 test images with 80 repetitions each. The raw event codes are one-based indices into the original sorted image split. Do not rename images/categories or substitute arbitrary THINGS image collections.

## EEG files

Each trusted local `train.pt` / `test.pt` dictionary contains:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `eeg` | `[N, R, C, T]`, tensor or NumPy array | Image condition, repetition, electrode, sample |
| `ch_names` | List of C strings | Electrode ordering; legacy 63-channel ordering is supported when absent |
| `img` | N strings or `[N, R]` strings | Image paths; every repetition of a condition must match |
| `label` | `[N]` or `[N, R]` | Category IDs, used for category-disjoint validation |
| `times` | `[T]` | Sample times from preprocessing; retained for interpretation |

The loader selects channels by name, crops sample indices, then averages repetitions (default). It also supports unaveraged trials and keeps all repetitions paired with the correct image. Validation splitting is by category, not by individual repetition. New preprocessing preserves float32 data; old NumPy caches may contain float16. Only load EEG pickle files you trust: NumPy-based legacy files require `torch.load(..., weights_only=False)`.

## Image feature files

The extractor writes:

| Key | Shape/type | Meaning |
| --- | --- | --- |
| `Latents` | `[N, 16, 1280]` | Intermediate resampler features |
| `token_importance` | `[N, 16]` | Final-layer attention received by latent keys |
| `image_ids` | N unique strings | `<category>/<filename>`; portable across machines |
| `topk_idx` | `[N, 16]` | Importance ranking, retained for inspection |
| `metadata` | Dictionary | Encoder/checkpoint/importance provenance |

Feature tensors must be finite; importance must be nonnegative with positive mass per image. Feature rows may be reordered because pairing uses `image_ids`. The ID is the final two path components, so original absolute EEG image paths are accepted without carrying machine-specific prefixes into the new caches.

Legacy feature files without `image_ids` are rejected by default. After independently checking that feature rows follow exactly the EEG condition order, set `data.allow_legacy_index_alignment: true` to use them. Missing importance cannot be inferred from `topk_idx`; regenerate those caches with the frozen feature extractor. Both split files use the `.pt` suffix.
