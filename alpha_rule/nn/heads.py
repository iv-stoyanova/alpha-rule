"""
Output heads for the dual-head AlphaZero-style network.

``PolicyHead``: a linear map to logits over the full vocabulary. The
``NeuralEvaluator`` masks non-applicable productions to ``-inf`` before
the softmax, so the head does not need to know which actions are legal.

``ValueHead``: a linear map then ``tanh`` to a scalar in ``(-1, +1)``.
The output range is fixed by design: the value target is
``clip(state_value / value_scale, -1, +1)`` (see
``Trajectory.value_targets`` and ``ReplayBuffer.value_scale``), and an
unbounded head would have to learn the scale itself, which interacts
badly with Adam's second-moment estimate on large-magnitude rewards.
Tanh saturates near the bounds, so genuinely extreme rewards lose
gradient signal at the asymptote, the standard AlphaZero tradeoff.
"""
from __future__ import annotations

import torch
from torch import nn


class PolicyHead(nn.Module):
    def __init__(self, d_model: int, num_productions: int):
        super().__init__()
        self.linear = nn.Linear(d_model, num_productions)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)                              # (B, num_productions)


class ValueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        # Zero-init so an untrained value head outputs 0 (tanh(0)) for every
        # input. The default init reads mildly positive at iteration 0, and that
        # uniform optimism gets backed up so the search latches onto whichever
        # node it first expands; starting at 0 keeps untrained leaves neutral and
        # lets the simulator signals drive the early search. Gradients still flow
        # once targets arrive (d tanh(Wh+b)/dW = h at W=b=0).
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(h)).squeeze(-1)      # (B,) in (-1, +1)
