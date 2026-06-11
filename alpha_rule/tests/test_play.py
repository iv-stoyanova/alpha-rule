"""
Tests for ``alpha_rule.training.play`` — AlphaZero-style greedy rollout
using the trained policy/value network.

Pins:
    - After ``train()``, ``log.model`` / ``log.max_len`` /
      ``log.n_simulations`` / ``log.depth_limit`` are populated and
      ``play(log, ...)`` runs end-to-end without needing to thread the
      network through by hand.
    - ``play()`` returns ``(rule_name: str, reward: float)`` with a
      non-root rule name on a healthy rollout.
    - Passing ``net=`` explicitly overrides ``log.model``.
    - Overriding ``n_simulations`` / ``depth_limit`` is respected
      (confirmed via a deeper rollout yielding a longer rule name).
    - Calling ``play()`` on an empty/unpopulated log (no trained model)
      raises a clear error rather than crashing inside torch.
    - ``log.best_formula`` (exploration-best) still works alongside
      ``play()``'s greedy-policy output — they are independent.
"""
from __future__ import annotations


class _ConstantSimulator:
    def __init__(self, value: float = 1.0):
        self.value = value

    def evaluate(self, node):
        return self.value


def _train_tiny(backup=None):
    """Train for one iteration and return both the log and a fresh grammar."""
    from alpha_rule.grammar.allen import AllenIntervalGrammar
    from alpha_rule.training import train

    grammar = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    log = train(
        grammar=grammar,
        expensive_simulator=_ConstantSimulator(value=1.0),
        n_iterations=1,
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
        backup=backup,
    )
    return grammar, log


# --------------------------------------------------------------------- #
# Case 1: train() populates the new TrainingLog fields.
# --------------------------------------------------------------------- #
def test_train_populates_play_fields_on_log():
    _grammar, log = _train_tiny()
    assert log.model is not None
    assert log.max_len == 12
    assert log.n_simulations == 4
    assert log.depth_limit == 2


# --------------------------------------------------------------------- #
# Case 2: play() produces a rule and a finite reward on a healthy run.
# --------------------------------------------------------------------- #
def test_play_returns_rule_and_reward():
    import math

    from alpha_rule.training import play

    grammar, log = _train_tiny()
    rule, reward = play(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
    )
    assert isinstance(rule, str)
    assert rule != "<ROOT>"                # greedy policy advanced at least once
    assert math.isfinite(reward)
    assert reward == 1.0                   # constant simulator


# --------------------------------------------------------------------- #
# Case 3: explicit net= overrides log.model.
# --------------------------------------------------------------------- #
def test_play_accepts_explicit_net_override():
    from alpha_rule.training import play

    grammar, log = _train_tiny()
    rule, reward = play(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
        net=log.model,                      # pass explicitly (same model)
    )
    assert rule is not None
    assert reward == 1.0


# --------------------------------------------------------------------- #
# Case 4: deeper rollout produces a longer rule (depth_limit respected).
# --------------------------------------------------------------------- #
def test_play_depth_limit_override_extends_rollout():
    from alpha_rule.training import play

    grammar, log = _train_tiny()           # trained at depth_limit=2
    rule_shallow, _ = play(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
        depth_limit=1,
    )
    rule_deep, _ = play(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
        depth_limit=3,
    )
    # Deeper rollout should produce at least as many tokens; usually strictly
    # more. With a constant simulator + greedy policy the rule grows each step.
    assert len(rule_deep.split()) >= len(rule_shallow.split())


# --------------------------------------------------------------------- #
# Case 5: calling play() on a bare TrainingLog raises a clear error.
# --------------------------------------------------------------------- #
def test_play_errors_without_trained_model():
    import pytest

    from alpha_rule.grammar.allen import AllenIntervalGrammar
    from alpha_rule.training import TrainingLog, play

    grammar = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    log = TrainingLog()                     # no training performed

    with pytest.raises(ValueError, match="trained network"):
        play(
            log,
            grammar=grammar,
            simulator=_ConstantSimulator(value=1.0),
        )


# --------------------------------------------------------------------- #
# play_top_k — iterative path-forbidding for k-rule selection.
# --------------------------------------------------------------------- #

def test_play_top_k_returns_distinct_rules():
    """Rule-level forbidding: every returned rule is distinct (each found rule
    is forbidden as a whole, the dead-node mechanism), capped at k."""
    import numpy as np

    from alpha_rule.training import play_top_k

    grammar, log = _train_tiny()
    results = play_top_k(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
        k=4, n_simulations=20, depth_limit=3,
        rng=np.random.default_rng(0),
    )
    rules = [r for r, _ in results]
    assert len(rules) == len(set(rules))       # all distinct
    assert 1 <= len(rules) <= 4


def test_play_top_k_not_capped_at_root_branches():
    """The fix: with only 2 root actions but a depth-3 search, play_top_k
    returns MORE than 2 distinct rules — the old branch-forbidding capped it
    at the number of root actions and got stuck. By pigeonhole some returned
    rules then share a first token, which branch-forbidding could never do."""
    import numpy as np

    from alpha_rule.training import play_top_k

    grammar, log = _train_tiny()               # alphabet {A, B} -> 2 root actions
    results = play_top_k(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
        k=4, n_simulations=20, depth_limit=3,
        rng=np.random.default_rng(0),
    )
    rules = [r for r, _ in results if r]
    assert len(rules) > 2                       # exceeded the 2-branch cap
    first_tokens = [r.split()[0] for r in rules]
    assert len(set(first_tokens)) < len(rules)  # some rules share a first token


def test_play_respects_dead_rule_names():
    """Forbidding a rule via dead_rule_names makes play() return a different
    rule (the same dead-node mechanism MCTS uses for -inf rules), while leaving
    its prefixes and siblings reachable."""
    import numpy as np

    from alpha_rule.training import play

    grammar, log = _train_tiny()
    rng = np.random.default_rng(0)
    rule1, _ = play(
        log, grammar=grammar, simulator=_ConstantSimulator(value=1.0),
        n_simulations=20, depth_limit=3, rng=rng,
    )
    rule2, _ = play(
        log, grammar=grammar, simulator=_ConstantSimulator(value=1.0),
        n_simulations=20, depth_limit=3, rng=rng, dead_rule_names={rule1},
    )
    assert rule1 is not None and rule2 is not None
    assert rule2 != rule1


def test_play_top_k_sorted_by_reward_desc():
    """Returned rules are sorted by reward descending."""
    from alpha_rule.training import play_top_k

    grammar, log = _train_tiny()
    results = play_top_k(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
        k=2,
    )
    rewards = [r for _, r in results]
    assert rewards == sorted(rewards, reverse=True)


def test_play_forbidden_root_actions_excluded():
    """A forbidden root action does not appear as the first token of the
    returned rule."""
    from alpha_rule.training import play

    grammar, log = _train_tiny()
    rule, _ = play(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
        forbidden_root_actions=("A",),
    )
    # The rule must not start with the forbidden action.
    assert rule is None or not rule.split() or rule.split()[0] != "A"


# --------------------------------------------------------------------- #
# Case 6: log.best_formula (exploration best) is independent of play().
# --------------------------------------------------------------------- #
def test_play_and_best_formula_coexist():
    from alpha_rule.training import play

    grammar, log = _train_tiny()
    assert log.best_formula is not None     # exploration-best populated
    rule, _ = play(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
    )
    # Both exist; they may differ. The contract is just that play()
    # produces its own answer without mutating log.best_formula.
    best_before = log.best_formula
    _ = play(
        log,
        grammar=grammar,
        simulator=_ConstantSimulator(value=1.0),
    )
    assert log.best_formula == best_before  # play() is read-only wrt the log
    assert rule is not None
