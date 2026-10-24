# RAHS

Official retrieval code for **[Region-Aware Hierarchical Sub-Feature Alignment for Robust EEG-Based Visual Decoding](https://openaccess.thecvf.com/content/CVPR2026F/html/Zhu_Region-Aware_Hierarchical_Sub-Feature_Alignment_for_Robust_EEG-Based_Visual_Decoding_CVPRF_2026_paper.html)**, CVPR 2026 Findings.

Yanan Zhu, Ziwei Xiang, Jiamin Wu, Jinyang Guo, Hongyuan Zhang, Chunfeng Song, Hongjian Fang, Yufei Guo, Xianglong Liu.

## Installation

Use Python 3.10 or newer. Install an appropriate [PyTorch](https://pytorch.org/get-started/locally/) build, then:

```bash
git clone git@github.com:aWeirdName/RAHS.git
cd RAHS
python -m pip install -e .
# Optional, for image feature extraction and raw EEG preprocessing:
python -m pip install -e '.[features,preprocess]'
```

## Training

```bash
# Same participant for training and final testing.
python -m rahs.train --config configs/rahs.yaml \
  --subject sub-01 --seed 100 --output outputs/sub-01_seed100

# Train on the other nine participants; test only on sub-01.
python -m rahs.train --config configs/rahs_inter_subject.yaml \
  --subject sub-01 --seed 100 --output outputs/loso_sub-01_seed100
```

## Evaluation

```bash
python -m rahs.evaluate \
  --checkpoint outputs/sub-01_seed100/best.pt \
  --output outputs/sub-01_seed100/evaluation.json \
  --export-features outputs/sub-01_seed100/retrieval_features.pt
```

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
tests/                 CPU unit and training/evaluation integration tests
```

## Validation

```bash
python -m unittest discover -s tests -v
```

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
