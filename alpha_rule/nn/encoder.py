"""
Small Transformer encoder for tokenised rule strings.

Architecture:
    embed(token_id) + pos(position)
        then N x TransformerEncoderLayer (batch_first)
        then a CLS-style learned pooler (Linear + Tanh) on the BOS hidden state

The pooled state is what the policy and value heads consume. The learned
projection (BERT-pooler pattern) decouples BOS's sequence-start role from
its pooling-source role: BOS still anchors position 0, but the heads see a
separately-projected vector. CPU-only by default; ``.to(device)`` works as
expected.
"""
from __future__ import annotations

import torch
from torch import nn


class FormulaEncoder(nn.Module):
    """Tokenised-formula encoder. Returns a single ``(B, d_model)`` vector."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=num_layers)
        # CLS-style learned pooler: BERT pattern. Linear projection +
        # Tanh keeps activations bounded so downstream heads see a
        # well-scaled signal.
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``LongTensor`` of shape ``(B, L)``, ``L <= max_len``.

        Returns:
            ``(B, d_model)`` pooled hidden state: the BOS token's
            contextualised representation passed through a learned
            CLS-style projection (``Linear + Tanh``).
        """
        B, L = x.shape
        if L > self.max_len:
            raise ValueError(f"Input length {L} > max_len {self.max_len}")
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.token_embed(x) + self.pos_embed(positions)
        h = self.trunk(h)
        return self.pooler(h[:, 0, :])                  # CLS-style pool
