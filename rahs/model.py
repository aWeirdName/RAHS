"""EEG encoder and hierarchical sub-feature projection.

Adapted from base/eeg_model_new.py in aWeirdName/iclr26_exp1.
Unused experimental branches were removed; active parameter names are retained.
"""

import math

import torch
from torch import nn


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class TemporalSpectralEmbedding(nn.Module):
    """Fuse a temporal projection, rFFT spectrum, and the EEG residual."""

    def __init__(self, time_samples=250, dropout=0.1, spectrum="power"):
        super().__init__()
        if spectrum not in {"power", "magnitude"}:
            raise ValueError("spectrum must be power or magnitude")
        self.spectrum = spectrum
        self.value_embedding = nn.Sequential(
            nn.Linear(time_samples, time_samples),
            nn.LayerNorm(time_samples),
            nn.Dropout(dropout),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(time_samples // 2 + 1, time_samples),
            nn.LayerNorm(time_samples),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, eeg):
        spectrum = torch.fft.rfft(eeg, dim=-1, norm="forward").abs()
        if self.spectrum == "power":
            spectrum = spectrum.square()
        return self.dropout(self.value_embedding(eeg) + self.spectral_proj(spectrum) + eeg)


class RAHSModel(nn.Module):
    """Map selected EEG channels to global and fine visual representations.

    Input: [batch, channels, time_samples]. Outputs: [batch, tokens * dim]
    and [batch, tokens, dim]. ``after`` is the original hierarchical ordering;
    the other orders are retained as explicitly configured ablations.
    """

    def __init__(self, channels=17, time_samples=250, num_tokens=16, dim=1280,
                 order="after", global_depth=3, fine_depth=3, residual=False,
                 dropout=0.1, spectrum="power", adapter_residual=True):
        super().__init__()
        if min(channels, time_samples, num_tokens, dim, global_depth, fine_depth) < 1:
            raise ValueError("Model dimensions and projection depths must be positive")
        if order not in {"after", "before", "branch", "mean"}:
            raise ValueError(f"Unknown projection order: {order}")
        self.channels, self.time_samples = channels, time_samples
        self.num_tokens, self.dim, self.order = num_tokens, dim, order
        self.adapter_residual = adapter_residual
        self.emb = TemporalSpectralEmbedding(time_samples, dropout, spectrum)
        self.fc = nn.Linear(channels * time_samples, num_tokens * dim)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        def projection(depth):
            blocks = []
            for _ in range(depth):
                block = nn.Sequential(nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim))
                blocks.append(ResidualAdd(block) if residual else block)
            return nn.Sequential(*blocks)

        self.prj = projection(global_depth)
        self.prj_f = projection(fine_depth)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.logit_scale1 = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, eeg):
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (self.channels, self.time_samples):
            raise ValueError(f"Expected [B, {self.channels}, {self.time_samples}], got {tuple(eeg.shape)}")
        base = self.fc(self.emb(eeg).flatten(1)).reshape(-1, self.num_tokens, self.dim)
        if self.order == "after":
            global_tokens = self.prj(base) + (base if self.adapter_residual else 0)
            fine_tokens = self.prj_f(global_tokens)
        elif self.order == "before":
            fine_tokens = self.prj_f(base)
            global_tokens = self.prj(fine_tokens) + (fine_tokens if self.adapter_residual else 0)
        elif self.order == "branch":
            global_tokens = self.prj(base) + (base if self.adapter_residual else 0)
            fine_tokens = self.prj_f(base)
        else:
            global_tokens = fine_tokens = (self.prj(base) + self.prj_f(base)) / 2
        return global_tokens.flatten(1), fine_tokens
