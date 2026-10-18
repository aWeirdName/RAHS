"""Global and importance-weighted sub-feature alignment objectives."""

import torch
from torch.nn import functional as F


def alignment_loss(model, eeg_global, eeg_fine, image_tokens, importance,
                   mode="topk", weight=0.1, topk=5, local_reduction="mean", image_indices=None):
    """Combine global contrastive alignment with Eq. (3) local alignment.

    The original global objective normalizes only image features. The optional ``token_contrastive``
    objective averages over tokens *after* normalizing their importance, hence
    includes a 1 / num_tokens factor. The top-k default averages selected terms as in Eq. (3); ``sum``
    reproduces the experimental pt_c scaling. Repeated image IDs can be
    supplied for multi-positive cross-subject global alignment.
    """
    if mode not in {"none", "token_contrastive", "topk"}:
        raise ValueError(f"Unknown alignment mode: {mode}")
    if weight < 0:
        raise ValueError("Alignment weight must be nonnegative")
    if image_tokens.shape != eeg_fine.shape:
        raise ValueError("Image and EEG sub-features must have identical [B, T, D] shapes")
    batch, tokens, _ = image_tokens.shape
    image_global = F.normalize(image_tokens.flatten(1).float(), dim=-1)
    logits = F.softplus(model.logit_scale) * eeg_global @ image_global.T
    labels = torch.arange(batch, device=logits.device)
    if image_indices is None:
        global_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    else:
        positives = image_indices[:, None] == image_indices[None, :]
        targets = positives.to(logits.dtype) / positives.sum(1, keepdim=True)
        global_loss = -(targets * (F.log_softmax(logits, dim=1) + F.log_softmax(logits.T, dim=1))).sum(1).mean() / 2
    fine_loss = global_loss.new_zeros(())
    if mode != "none":
        if importance.shape != (batch, tokens):
            raise ValueError("Importance must have shape [B, T]")
        eeg = F.normalize(eeg_fine, dim=-1)
        image = F.normalize(image_tokens.float(), dim=-1)
        if mode == "token_contrastive":
            weights = importance.mean(0)
            weights = weights / (weights.sum() + 1e-8)
            scores = torch.einsum("btd,ctd->tbc", eeg, image)
            scores = scores * F.softplus(model.logit_scale1)
            targets = labels.expand(tokens, -1).reshape(-1)
            forward = F.cross_entropy(scores.reshape(-1, batch), targets, reduction="none")
            backward = F.cross_entropy(scores.transpose(1, 2).reshape(-1, batch), targets, reduction="none")
            fine_loss = (((forward + backward) / 2).reshape(tokens, batch) * weights[:, None]).mean()
        else:
            if not 1 <= topk <= tokens:
                raise ValueError("topk must be between 1 and num_tokens")
            indices = importance.topk(topk, dim=1).indices
            weighted = (eeg * image).sum(-1) * importance
            fine_loss = -weighted.gather(1, indices).sum(1).mean()
            if local_reduction == "mean":
                fine_loss = fine_loss / topk
            elif local_reduction != "sum":
                raise ValueError("local_reduction must be mean or sum")
    return global_loss + weight * fine_loss, {"global_loss": global_loss.detach(), "fine_loss": fine_loss.detach()}
