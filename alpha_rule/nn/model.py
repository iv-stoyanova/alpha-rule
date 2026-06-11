"""
``AllenFormulaNet``: encoder plus (policy, value) heads.

The model is intentionally small (~tens of thousands of parameters at
default config) since the MVP runs on CPU and search dominates the
wall-clock budget. Larger configs are a one-line change.

The tokeniser is held on the model so call-sites only thread one object
around. ``num_productions`` defaults to ``tokenizer.vocab_size()`` so
the policy head's output index space lines up with the tokeniser's
token id space, so ``priors[id_of["A"]]`` is the prior probability
of producing token ``"A"``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from alpha_rule.nn.encoder import FormulaEncoder
from alpha_rule.nn.heads import PolicyHead, ValueHead
from alpha_rule.nn.tokenizer import GrammarTokenizer


class AllenFormulaNet(nn.Module):
    """Tokeniser + encoder + (policy, value) heads."""

    def __init__(
        self,
        tokenizer: GrammarTokenizer,
        *,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        max_len: int = 256,
        dropout: float = 0.0,
        num_productions: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encoder = FormulaEncoder(
            vocab_size=tokenizer.vocab_size(),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            max_len=max_len,
            dropout=dropout,
            pad_id=tokenizer.pad_id,
        )
        self.policy = PolicyHead(d_model, num_productions or tokenizer.vocab_size())
        self.value = ValueHead(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``(B, L)`` LongTensor (tokenised rule strings).

        Returns:
            ``(policy_logits, value)`` of shapes ``(B, num_productions)``
            and ``(B,)`` respectively. Logits are unmasked; masking
            happens in ``NeuralEvaluator`` once we know which
            productions are applicable to the specific state.
        """
        h = self.encoder(x)
        return self.policy(h), self.value(h)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gradient-free inference forward (eval mode, no autograd) for scoring
        nodes during search -- the search never needs gradients, and this
        avoids building the autograd graph on the per-node hot path. Restores
        the prior train/eval mode on return.

        The eval/train switch is skipped when the model is already in eval
        mode. ``nn.Module.eval()`` and ``.train()`` each recursively walk every
        submodule; on the per-node search hot path that toggle costs more than
        the forward itself. Keeping the net in eval outside the gradient step
        (``train_step`` flips to train, then restores eval) makes this a no-op."""
        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.inference_mode():
                return self.forward(x)
        finally:
            if was_training:
                self.train(True)
