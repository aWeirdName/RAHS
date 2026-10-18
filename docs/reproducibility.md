# Implementation and reproducibility

## Sources

- Published paper: [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Zhu_Region-Aware_Hierarchical_Sub-Feature_Alignment_for_Robust_EEG-Based_Visual_Decoding_CVPRF_2026_paper.pdf), especially Sections 3.2–3.5 and 4.1–4.2.
- Experimental code: `aWeirdName/iclr26_exp1`, commit `b9c8eb4cffe0d95440496b1c55ed3efba345b71a`.
- Main reference: `main_new.py`, `base/eeg_model_new.py`, `base/data_eeg.py`, `emb_gen/resamper.py`, and `emb_gen/gen.py`. `main_new_exp.py` differs mainly in feature-export paths; temporal-window and channel ablations are not separate public training programs.

## Scope

The author requested a retrieval-only release. The objective is `L_global + lambda_local * L_local`, with the reconstruction term omitted. No diffusion model, generation code, or full-paper performance claim is included. Released configurations are runnable reference settings; the source repository does not identify a uniquely verified final training command/checkpoint.

## Explicit differences

| Component | Experimental implementation | Standard released configuration |
| --- | --- | --- |
| Channel spectrum | `abs(rfft(x, norm="forward"))` | Squared magnitude, matching the power-spectrum description in Section 3.2 |
| Domain adapter | Three sequential LN/GELU/Linear blocks; no residual by default | Same MLP with an outer residual `z + g(z)`, as described in Section 3.2 |
| Hierarchy | Configurable; `after` default | Global adapter followed by local projector |
| Local alignment | Optional `pt` or `pt_c`; disabled in CLI defaults | Top-k weighted cosine (`pt_c` family), averaged over k as in Eq. (3) |
| Local loss scale | `pt_c` sums k weighted similarities | Mean over selected k terms |
| Global logits | Image features normalized; EEG features unnormalized; softplus scale | Retained from source; retrieval normalizes both sides |
| Intra-subject selection | Test loader also used as validation loader | Hold out whole categories from training; final test after selection |
| Inter-subject selection | Test images from training participants used for validation | Training-category holdout among the non-target participants |
| Repeated image IDs in training | Diagonal-only paired CE | Uniform probability over same-image positives, including across subjects |
| Retrieval | Candidate set is the current minibatch | Complete unique-image bank, with chunked queries |
| Feature pairing | Positional row index | Explicit category/filename IDs; repeat index handled correctly |
| Feature cache | Generator omits `token_importance` required by loader | Importance and image manifest always saved |
| Preprocessing storage | Half-precision NumPy arrays and full epoch times | Float32 tensors and exact retained sample times |

These are deliberate, documented differences, not evidence that the original results used the new settings. `configs/legacy_experiment.yaml` restores magnitude spectra, nonresidual adapters, summed top-k loss, diagonal CE, and test-based model selection for the intra-subject `pt_c` experiment. Runtime fixes (portable paths, manifest validation, full candidate banks) remain. Removing unused layers changes random-number consumption during initialization, so even legacy settings are not a bitwise rerun of an old seed.

Other architecture/loss variants are available through config only: `model.order` (`after`, `before`, `branch`, `mean`), projection depths, block residuals, and `alignment.mode` (`topk`, `token_contrastive`, `none`). The `token_contrastive` mode preserves the experimental batch-averaged importance weighting and extra `1 / num_tokens` scaling; it is not the local loss in Eq. (3). For duplicate-image minibatches this ablation retains a diagonal-only token objective even when the global objective uses multiple positives.

## Hyperparameter provenance

The paper identifies THINGS-EEG, a ViT-H/14 + pretrained IP-Adapter resampler, and both intra-/inter-subject evaluation. Its main text does not supply a complete executable training configuration. The defaults below come from code or explicit release choices, not an inferred final result:

- 16 tokens of dimension 1280; 17 posterior channels; 250 EEG samples: active model/data configuration.
- Three blocks per projection, dropout 0.1, AdamW, learning rate `1e-4`, weight decay `1e-4`, 40 epochs, seed 100: active source defaults.
- Local weight 0.1 and top-k 5: CLI defaults, with top-k alignment explicitly enabled in the release.
- Train batch size 1024, test query batch size 200: source configuration.
- Early stopping: patience 10 intra-subject / 5 inter-subject, improvement threshold 0.001: source behavior.
- Category holdout fraction 0.1 and multiple-positive global loss: release choices to separate selection from testing and avoid treating identical image targets as negatives.

Changing from summed top-k loss to its Eq. (3) mean changes its effective strength. To reproduce the source's weighted local term while using mean reduction, multiply the source local weight by k. The default does not apply this conversion silently.

## Attention weights and visual targets

The vision encoder contributes its penultimate hidden state. The frozen resampler returns 16 intermediate 1280-dimensional latents **before** `proj_out` and `norm_out`. These are the training targets; projected diffusion-conditioning vectors are not used.

The original attention recipe averages the last resampler layer over heads and query positions, then selects the last 16 key positions (the latent-key block). These raw nonnegative weights are retained, not renormalized for top-k cosine alignment. The resampler and pretrained `image_proj` state-dict names remain compatible with IP-Adapter Plus SDXL ViT-H weights.

## Evaluation and artifacts

Retrieval uses cosine similarity across flattened global features and a unique candidate image bank. Ranking ties are resolved by candidate order. Top-5 uses `min(5, number_of_candidates)` for small synthetic datasets. Metrics JSON includes candidate/query counts, the checkpoint-selection protocol and selected epoch.

In standard inter-subject runs the held-out participant contributes no training or validation EEG. All non-target participants share a feature bank, while training/validation categories stay disjoint across participants. This is single-device training; distributed/multi-GPU behavior is not advertised.

`best.pt` and `last.pt` are inference checkpoints with the resolved config; training resume is not implemented. Original Lightning checkpoints contain unused VIB/convolution/embedding parameters and require an explicit conversion rather than loading them with `strict=False` and ignoring mismatches.

## Validation limits

Synthetic CPU tests check numerical formulas, tensor shapes, gradients, pairing, subject/category exclusion and training/checkpoint/evaluation round-trips. They are not a substitute for training on the original THINGS-EEG data. No full-size real-data accuracy, GPU benchmark, pretrained feature-extraction run or raw-recording preprocessing run is claimed for this release.
