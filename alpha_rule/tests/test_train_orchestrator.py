"""
Smoke test for ``alpha_rule.training.train`` — the outer self-play +
NN-update loop wiring everything together.

Pins:
    - 3 iterations on a tiny grammar with a constant-reward simulator
      complete without raising.
    - ``TrainingLog`` accumulates per-iteration metrics with finite
      losses (or 0 for warmup iterations).
    - ``best_reward`` tracks the highest finite reward observed.

This is intentionally a smoke check, not a convergence assertion: the
random seeds and tiny network make per-iteration loss noisy. What we
care about is that all of Phases A–G interlock cleanly under the
``train`` entry point.
"""
from __future__ import annotations

import math


class _ConstantSimulator:
    def __init__(self, value: float = 1.0):
        self.value = value
        self.calls = 0

    def evaluate(self, node):
        self.calls += 1
        return self.value


def test_train_runs_three_iterations_without_crashing():
    from alpha_rule.grammar.allen import AllenIntervalGrammar
    from alpha_rule.training import train

    grammar = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    log = train(
        grammar=grammar,
        expensive_simulator=_ConstantSimulator(value=1.0),
        n_iterations=3,
        n_simulations=4,
        depth_limit=2,
        buffer_capacity=64,
        buffer_warmup=1,
        batch_size=4,
        train_steps_per_iteration=1,
        d_model=16,
        nhead=2,
        num_layers=1,
        max_len=12,
        learning_rate=1e-2,
        seed=0,
    )
    assert len(log.iterations) == 3


def test_train_accepts_custom_selection_and_backup_pairing():
    """``train()`` exposes ``selection`` and ``backup`` so callers can
    pair non-default strategies (e.g. PercentileRewardBackup with
    PUCTSelection(q_source='filtered_mean'))."""
    from alpha_rule.grammar.allen import AllenIntervalGrammar
    from alpha_rule.mcts.backprop import PercentileRewardBackup
    from alpha_rule.mcts.selection import PUCTSelection
    from alpha_rule.training import train

    class _Recording(PUCTSelection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def select(self, parent):
            self.calls += 1
            return super().select(parent)

    grammar = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    selection = _Recording(c_puct=1.0, q_source="filtered_mean")
    backup = PercentileRewardBackup(percentile=20, min_samples=2)

    log = train(
        grammar=grammar,
        expensive_simulator=_ConstantSimulator(value=1.0),
        n_iterations=2,
        n_simulations=4,
        depth_limit=2,
        buffer_capacity=32,
        buffer_warmup=1,
        batch_size=4,
        train_steps_per_iteration=1,
        d_model=16,
        nhead=2,
        num_layers=1,
        max_len=12,
        learning_rate=1e-2,
        seed=0,
        selection=selection,
        backup=backup,
    )
    assert len(log.iterations) == 2
    # The custom selection's ``select`` method must have actually been
    # called during the run — confirms wiring, not just storage.
    assert selection.calls > 0
    for it in log.iterations:
        assert math.isfinite(it.train_total)
        assert math.isfinite(it.train_policy)
        assert math.isfinite(it.train_value)
    # Best reward should equal the constant value (1.0) after at least
    # one trajectory step succeeded.
    assert log.best_reward == 1.0
    assert log.best_formula is not None
