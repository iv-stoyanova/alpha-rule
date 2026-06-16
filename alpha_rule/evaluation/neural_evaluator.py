"""
``NeuralEvaluator``: wraps an ``AllenFormulaNet`` as an ``Evaluator``.

One inference pass per call. Returns ``EvalResult(value, priors)`` where
``priors`` maps each applicable production name to a softmaxed probability over
only the legal actions (illegal logits are masked to ``-inf`` before the
softmax).

The model's value head is ``tanh``-bounded to ``(-1, +1)``. This evaluator
multiplies the raw output by ``value_scale`` to recover raw-reward units, so
``value_scale`` should match the ``value_scale`` used at training time
(``run_self_play`` / ``ReplayBuffer``, i.e. the simulator's positive reward
cap). The default ``1.0`` returns the network's value unchanged.

Plug-compatible with ``RuleSimulator`` (both implement ``Evaluator``) but cheap:
no environment episodes, just a forward pass. A typical loop scores leaves with
the network and only spends the expensive simulator on the chosen step.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from alpha_rule.evaluation.evaluator import EvalResult, Evaluator
from alpha_rule.grammar.grammar import Grammar

if TYPE_CHECKING:
    from alpha_rule.nn.model import AllenFormulaNet


DEFAULT_VALUE_SCALE: float = 1.0
"""Default multiplier on the network's tanh-bounded value output: ``1.0``, so
the value is returned unchanged in ``(-1, +1)``. Pass the training-time
``value_scale`` (the simulator's positive reward cap) to recover raw-reward
units for MCTS backup."""


class NeuralEvaluator(Evaluator):
    """
    Args:
        model: trained or freshly-initialised ``AllenFormulaNet``.
        grammar: the grammar whose productions the priors are over. Used to
            figure out which logits are legal at each call.
        max_len: ``encode`` pads to this length. Should match the model's
            ``max_len`` so the position embeddings line up, and be large enough
            for the deepest rule (``encode`` raises otherwise).
        value_scale: multiplier applied to the network's raw value output before
            it is returned as ``EvalResult.value``. Default ``1.0`` (raw
            passthrough). Set it to the ``value_scale`` used at training time to
            recover raw-reward units.
    """

    def __init__(
        self,
        model: "AllenFormulaNet",
        grammar: Grammar,
        *,
        max_len: int,
        value_scale: float = DEFAULT_VALUE_SCALE,
        neg_value_scale: Optional[float] = None,
    ):
        if not (value_scale > 0):
            raise ValueError(
                f"value_scale must be > 0, got {value_scale!r}"
            )
        if neg_value_scale is not None and not (neg_value_scale > 0):
            raise ValueError(
                f"neg_value_scale must be > 0, got {neg_value_scale!r}"
            )
        self.model = model
        # Inference wrapper: keep the net in eval mode so per-node predict()
        # calls skip the recursive train/eval toggle. train_step() flips to
        # train for the gradient step and restores eval afterwards.
        self.model.eval()
        self.grammar = grammar
        self.max_len = max_len
        self.value_scale = float(value_scale)
        # Negative-side scale for asymmetric value de-scaling, matching the
        # ``neg_value_scale`` the replay buffer used when building targets. When
        # ``None`` it mirrors ``value_scale`` (symmetric, historical behaviour).
        self.neg_value_scale = (
            float(neg_value_scale) if neg_value_scale is not None else self.value_scale
        )

    @classmethod
    def from_simulator(
        cls,
        model: "AllenFormulaNet",
        grammar: Grammar,
        simulator,
        *,
        max_len: int,
        neg_value_scale: Optional[float] = None,
    ) -> "NeuralEvaluator":
        """Build an evaluator whose ``value_scale`` matches the simulator's
        ``reward_scale`` (the same cap ``run_self_play`` / ``ReplayBuffer`` use),
        so the network value is returned in raw-reward units consistent with the
        simulator's rewards in the shared MCTS backup. Prefer this over the bare
        constructor when wiring the net into search: the constructor defaults
        ``value_scale=1.0``, which silently mismatches a simulator whose
        ``reward_scale`` is not 1. Falls back to ``1.0`` if the simulator
        exposes no ``reward_scale``. ``neg_value_scale`` is passed through for
        asymmetric de-scaling (``None`` -> symmetric).
        """
        scale = getattr(simulator, "reward_scale", None) or DEFAULT_VALUE_SCALE
        return cls(
            model, grammar, max_len=max_len,
            value_scale=scale, neg_value_scale=neg_value_scale,
        )

    def evaluate(self, node) -> EvalResult:
        ids = self.model.tokenizer.encode(node.name, max_len=self.max_len).unsqueeze(0)
        ids = ids.to(next(self.model.parameters()).device)
        # predict() runs in eval + inference_mode and restores the model's prior
        # train/eval mode, so scoring a node never leaves the model in eval mode.
        logits, value = self.model.predict(ids)

        applicable = self.grammar.applicable_productions(node)
        prior_logits = logits.squeeze(0)            # (vocab_size,)
        mask = torch.full_like(prior_logits, float("-inf"))
        for prod in applicable:
            mask[self.model.tokenizer.id_of[prod.name]] = 0.0
        priors_full = F.softmax(prior_logits + mask, dim=-1)

        # Pull priors to CPU once: reading N entries via ``.item()`` on CUDA
        # would force N host syncs.
        priors_cpu = priors_full.detach().cpu().tolist()
        priors_dict = {
            prod.name: float(priors_cpu[self.model.tokenizer.id_of[prod.name]])
            for prod in applicable
        }
        # De-scale the tanh output back to reward units. Asymmetric when
        # neg_value_scale differs from value_scale (matching how the replay
        # buffer built the target); identical to ``value * value_scale`` when
        # they are equal (the symmetric default).
        z = float(value.item())
        raw = z * self.value_scale if z >= 0 else z * self.neg_value_scale
        return EvalResult(value=raw, priors=priors_dict)
