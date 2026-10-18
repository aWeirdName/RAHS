"""Retrieval against a complete candidate bank, independent of query batching."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def retrieval_metrics(queries, candidates, targets, chunk_size=256):
    if queries.ndim != 2 or candidates.ndim != 2 or queries.shape[1] != candidates.shape[1]:
        raise ValueError("Queries and candidates must be [N, D] and [M, D]")
    if len(queries) == 0 or len(candidates) == 0 or chunk_size < 1:
        raise ValueError("Retrieval requires nonempty features and a positive chunk size")
    targets = torch.as_tensor(targets, device=queries.device, dtype=torch.long)
    if targets.shape != (len(queries),) or targets.min() < 0 or targets.max() >= len(candidates):
        raise ValueError("Each query must identify one candidate")
    bank = F.normalize(candidates.to(queries.device).float(), dim=-1)
    ranks, similarities = [], []
    for start in range(0, len(queries), chunk_size):
        scores = F.normalize(queries[start:start + chunk_size].float(), dim=-1) @ bank.T
        truth = targets[start:start + chunk_size]
        # Stable sorting gives deterministic candidate-index ordering for ties.
        ordering = scores.argsort(dim=1, descending=True, stable=True)
        ranks.append((ordering == truth[:, None]).to(torch.int64).argmax(1) + 1)
        similarities.append(scores.gather(1, truth[:, None]).squeeze(1))
    rank = torch.cat(ranks).float()
    return {"top1": (rank == 1).float().mean().item(),
            "top5": (rank <= min(5, len(bank))).float().mean().item(),
            "mrr": rank.reciprocal().mean().item(),
            "matched_similarity": torch.cat(similarities).mean().item(),
            "num_queries": len(queries), "num_candidates": len(bank)}
