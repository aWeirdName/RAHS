# RAHS

Official retrieval code for **[Region-Aware Hierarchical Sub-Feature Alignment for Robust EEG-Based Visual Decoding](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhu_Region-Aware_Hierarchical_Sub-Feature_Alignment_for_Robust_EEG-Based_Visual_Decoding_CVPRF_2026_paper.html)**, CVPR 2026 Findings.

Yanan Zhu, Ziwei Xiang, Jiamin Wu, Jinyang Guo, Hongyuan Zhang, Chunfeng Song, Hongjian Fang, Yufei Guo, Xianglong Liu.

RAHS aligns channel-wise EEG representations with 16 visual sub-features extracted by a frozen CLIP ViT-H/14 encoder and IP-Adapter resampler. A global contrastive objective and attention-weighted top-k local objective train the hierarchical EEG projections.

**Release scope:** EEG preprocessing, image sub-feature extraction, intra-subject and leave-one-subject-out training, and image retrieval. Diffusion priors, reconstruction losses, and image generation are excluded. This release therefore does **not** implement the full reconstruction-assisted objective or claim reproduction of the paper's reported scores. The distinction between the paper equations, the experimental code, and the released defaults is documented in [Reproducibility](docs/reproducibility.md).

## Installation

Use Python 3.10 or newer. Install an appropriate [PyTorch](https://pytorch.org/get-started/locally/) build, then:

```bash
git clone git@github.com:aWeirdName/RAHS.git
cd RAHS
python -m pip install -e .
# Optional, for image feature extraction and raw EEG preprocessing:
python -m pip install -e '.[features,preprocess]'
```

Core CPU tests were run with Python 3.10, PyTorch 2.4.1 and NumPy 1.24.4. A GPU is recommended for the full-size model. No model or dataset downloads occur during import or training on cached features.

## Data preparation

Download the [THINGS-EEG recordings](https://osf.io/anp5v/) and corresponding [THINGS images](https://osf.io/jum2f/). Organize the EEG-associated image splits as `training_images/<category>/<image>` and `test_images/<category>/<image>`; preserve their original category/filename ordering. See [Data format](docs/data.md) for existing preprocessed files and manifests.

```bash
python -m rahs.preprocess \
  --raw-dir data/things-eeg/Raw_data \
  --image-dir data/things-eeg/Image_set \
  --output data/things-eeg/Preprocessed_data_250Hz_whiten \
  --subjects sub-01
```

Omit `--subjects` to process all ten participants. Whitening statistics are estimated from each session's training recordings only. The sample window matches the experimental preprocessing: the last 250 samples of the resampled `[-0.2, 1.0]` second epoch. It is not silently relabeled as an exact `[0, 1)` time grid.

For image features, download `models/image_encoder/` and `sdxl_models/ip-adapter-plus_sdxl_vit-h.bin` from [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter). Only the CLIP encoder and image resampler are loaded; a diffusion pipeline is not required.

```bash
python -m rahs.extract_features \
  --images data/things-eeg/Image_set/training_images \
  --image-encoder checkpoints/IP-Adapter/models/image_encoder \
  --adapter-checkpoint checkpoints/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin \
  --output data/things-eeg/image_features/image_latent_train.pt

python -m rahs.extract_features \
  --images data/things-eeg/Image_set/test_images \
  --image-encoder checkpoints/IP-Adapter/models/image_encoder \
  --adapter-checkpoint checkpoints/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin \
  --output data/things-eeg/image_features/image_latent_test.pt
```

The feature cache includes sub-features, attention importance, and image IDs. EEG/image pairs are matched by ID rather than assumed row ordering.

## Training

```bash
# Same participant for training and final testing.
python -m rahs.train --config configs/rahs.yaml \
  --subject sub-01 --seed 100 --output outputs/sub-01_seed100

# Train on the other nine participants; test only on sub-01.
python -m rahs.train --config configs/rahs_inter_subject.yaml \
  --subject sub-01 --seed 100 --output outputs/loso_sub-01_seed100
```

Use `--eeg-dir`, `--feature-dir`, and `--device` to override paths/device. All other settings are explicit in YAML. Repeated EEG trials are averaged by default. The standard configurations reserve 10% of **training categories** for validation, keeping all images/repetitions of each category together. The test split is evaluated after checkpoint selection.

Each run writes `config.yaml`, `split.json`, `history.jsonl`, `best.pt`, `last.pt`, and `metrics.json`. Existing run directories are never overwritten. Checkpoints contain model weights and configuration for inference, not optimizer/RNG state for training resume.

For compatibility investigations, `configs/legacy_experiment.yaml` matches the active `main_new.py --pt_mode pt_c` conventions. It explicitly selects checkpoints on test data and must not be reported as an independent held-out evaluation. It is not a verified paper-result recipe.

## Evaluation

```bash
python -m rahs.evaluate \
  --checkpoint outputs/sub-01_seed100/best.pt \
  --output outputs/sub-01_seed100/evaluation.json \
  --export-features outputs/sub-01_seed100/retrieval_features.pt
```

Retrieval ranks every query against the complete unique-image candidate bank: 200 images for the standard THINGS-EEG test split. Query batch size does not change the candidate set. Metrics are Top-1, Top-5, mean reciprocal rank (MRR), and matched-pair cosine similarity. With one relevant candidate per query, MRR equals the original implementation's `mAP`; it is labeled precisely here.

## Repository layout

```text
rahs/
  model.py             Channel embedder, fusion projector, hierarchical adapters
  losses.py            Global contrastive and adaptive local objectives
  data.py              Validated EEG/feature loading and category-level splits
  engine.py            Training and complete-bank retrieval
  train.py             Training command
  evaluate.py          Evaluation and feature export
  extract_features.py  Frozen visual sub-features and attention importance
  resampler.py         IP-Adapter-compatible visual resampler
  preprocess.py        Raw EEG epoching and training-only whitening
configs/               Explicit intra-subject, inter-subject, legacy settings
docs/                  Data contract, provenance, and reproduction limits
tests/                 CPU unit and training/evaluation integration tests
```

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic tensors and cover the loss equation, gradients, feature manifests, repetition alignment, category separation, complete-bank retrieval, subject exclusion, and checkpoint round-trips. They do not establish real-data accuracy. Full training, raw-recording preprocessing, and pretrained visual extraction require external datasets/weights.

## Citation

```bibtex
@InProceedings{Zhu_2026_CVPR,
  author = {Zhu, Yanan and Xiang, Ziwei and Wu, Jiamin and Guo, Jinyang and Zhang, Hongyuan and Song, Chunfeng and Fang, Hongjian and Guo, Yufei and Liu, Xianglong},
  title = {Region-Aware Hierarchical Sub-Feature Alignment for Robust EEG-Based Visual Decoding},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  month = {June},
  year = {2026},
  pages = {6667-6676}
}
```

## Acknowledgments and license

This release was organized from [aWeirdName/iclr26_exp1](https://github.com/aWeirdName/iclr26_exp1), which builds on **UBP: Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior** (CVPR 2025). We acknowledge [NICE-EEG](https://github.com/eeyhsong/NICE-EEG), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), and the THINGS/THINGS-EEG dataset authors. See [NOTICE](NOTICE) for provenance.

Code is distributed under the retained [Apache-2.0 license](LICENSE). Datasets and pretrained weights are obtained separately and retain their respective licenses.
