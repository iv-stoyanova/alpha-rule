"""
Training-step utilities for ``AllenFormulaNet``.

``train_step`` performs one gradient update on a minibatch of replay
rows. Rows come in two shapes:

    3-tuple (no mask):  (state, visit_pi, value_target)
    4-tuple (masked):   (state, visit_pi, value_target, applicable_actions)

The 4th element pins which production names are legal at ``state``.
When present it drives a softmax mask that mirrors
``NeuralEvaluator``'s inference path (illegal logits set to ``-inf``
before the softmax). When absent the loss uses an unmasked full-vocab
softmax.

Loss = MSE(value_pred, target) + cross_entropy(policy_pred, visit_pi).

The policy target is a probability distribution (visit fractions) over
productions; cross-entropy uses the soft-label form
``-sum(p_target * log_softmax(logits))``. Non-applicable productions
absent from ``visit_pi`` get implicit probability zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from alpha_rule.nn.model import AllenFormulaNet


@dataclass
class TrainStepLog:
    total: float
    policy: float
    value: float


def collate(
    batch: Iterable[Tuple],
    model: AllenFormulaNet,
    *,
    max_len: int,
):
    """
    Pack a list of replay rows into tensors. Accepts both 3-tuple rows
    ``(state, visit_pi, value_target)`` and 4-tuple rows
    ``(state, visit_pi, value_target, applicable_actions)``.

    ``state`` is converted to a string via ``.name`` if it's an
    ``MCTSRuleNode``; otherwise it must already be a string.

    ``visit_pi`` is a dict ``production_name -> probability``. Missing
    productions get probability zero.

    ``applicable_actions`` (4-tuple only) is an iterable of production
    names that are legal at ``state``. Drives a ``(B, vocab_size)``
    boolean mask used to restrict the softmax denominator at train time.

    Returns:
        ``(states_ids, target_pi, target_z)``                  if every
            row is a 3-tuple (no mask), OR
        ``(states_ids, target_pi, target_z, applicable_mask)`` if every
            row is a 4-tuple. Mixed batches are not supported.
    """
    rows = list(batch)
    if not rows:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            torch.empty((0,), dtype=torch.float32),
        )

    has_mask = len(rows[0]) == 4
    if has_mask and not all(len(r) == 4 for r in rows):
        raise ValueError(
            "collate: mixed 3-tuple and 4-tuple rows in the same batch"
        )

    states_ids = []
    pi_targets = []
    z_targets = []
    masks: List[torch.Tensor] = []
    vocab_size = model.tokenizer.vocab_size()
    for row in rows:
        if has_mask:
            state, visit_pi, z, applicable = row
        else:
            state, visit_pi, z = row
            applicable = None
        s_name = state.name if hasattr(state, "name") else str(state)
        states_ids.append(model.tokenizer.encode(s_name, max_len=max_len))
        pi_vec = torch.zeros(vocab_size, dtype=torch.float32)
        for tok, p in visit_pi.items():
            tok_id = model.tokenizer.id_of.get(tok)
            if tok_id is not None:
                pi_vec[tok_id] = float(p)
        pi_targets.append(pi_vec)
        z_targets.append(float(z))
        if has_mask:
            mask = torch.zeros(vocab_size, dtype=torch.bool)
            for name in applicable:
                tok_id = model.tokenizer.id_of.get(name)
                if tok_id is not None:
                    mask[tok_id] = True
            masks.append(mask)

    # Renormalise every policy-target row once (vectorised) in case a visit_pi
    # dict didn't sum to 1. All-zero rows stay all zeros.
    target_pi = torch.stack(pi_targets, dim=0)
    row_sums = target_pi.sum(dim=-1, keepdim=True)
    target_pi = target_pi / torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))

    out = (
        torch.stack(states_ids, dim=0),
        target_pi,
        torch.tensor(z_targets, dtype=torch.float32),
    )
    if has_mask:
        out = out + (torch.stack(masks, dim=0),)
    return out


def train_step(
    model: AllenFormulaNet,
    optimizer: torch.optim.Optimizer,
    batch: List[Tuple],
    *,
    max_len: int,
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
    grad_clip: float = 0.0,
) -> TrainStepLog:
    """
    Single training step. Returns a ``TrainStepLog`` with the three
    scalar losses.

    Args:
        grad_clip: if > 0, clip the global gradient L2-norm to this value
            before ``optimizer.step()``. Default ``0.0`` means no clipping.
            A modest positive value (e.g. ``1.0``) helps runs that ingest
            unnormalised large-magnitude value targets, where Adam's
            second-moment estimate can otherwise be destabilised.
    """
    if not batch:
        return TrainStepLog(0.0, 0.0, 0.0)
    out = collate(batch, model, max_len=max_len)
    if len(out) == 4:
        states_ids, target_pi, target_z, applicable_mask = out
    else:
        states_ids, target_pi, target_z = out
        applicable_mask = None

    # Move tensors to wherever the model lives (CPU / CUDA).
    device = next(model.parameters()).device
    states_ids = states_ids.to(device)
    target_pi = target_pi.to(device)
    target_z = target_z.to(device)
    if applicable_mask is not None:
        applicable_mask = applicable_mask.to(device)

    model.train()
    optimizer.zero_grad()
    logits, values = model(states_ids)
    if applicable_mask is not None:
        # Mirror NeuralEvaluator's inference-time softmax: inapplicable
        # logits go to -inf so the softmax denominator only sums over
        # legal productions. Without this, train- and inference-time
        # distributions disagree on the denominator.
        logits = logits.masked_fill(~applicable_mask, float("-inf"))
    log_probs = F.log_softmax(logits, dim=-1)
    if applicable_mask is not None:
        # ``target_pi`` is 0 at masked positions and ``log_probs`` is -inf
        # there. The product would be 0 * -inf = NaN. Zero the log-probs
        # outside the mask so the contribution is exactly 0.
        log_probs = torch.where(
            applicable_mask, log_probs, torch.zeros_like(log_probs)
        )
    policy_loss = -(target_pi * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, target_z)
    total = policy_weight * policy_loss + value_weight * value_loss
    total.backward()
    if grad_clip and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
    optimizer.step()
    # Restore eval mode: the gradient step is the only place the net needs
    # train mode. Leaving it in eval means search-time predict() calls skip the
    # recursive train/eval toggle on every scored node (a measured hot path).
    model.eval()
    return TrainStepLog(
        total=float(total.detach().item()),
        policy=float(policy_loss.detach().item()),
        value=float(value_loss.detach().item()),
    )


def default_optimizer(
    model: AllenFormulaNet,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> torch.optim.Optimizer:
    """Adam with AlphaZero-style L2 weight regularisation.

    The AlphaGo-Zero / AlphaZero loss includes a ``c * ||theta||^2`` term;
    ``train_step`` does not add it to the loss, so it is applied here via the
    optimiser's ``weight_decay`` (the standard PyTorch way). Pass the result
    as the ``optimizer`` argument to ``train_step``.
    """
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
