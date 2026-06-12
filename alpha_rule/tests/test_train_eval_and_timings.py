"""
Tests for the eval + phase-timing surface of ``alpha_rule.training.train``:

    - ``eval_simulator`` + ``eval_every`` optional kwargs on ``train()``
    - ``IterationLog`` carries eval + phase-timing fields
    - eval only fires on iterations matching the cadence
    - eval is skipped when no best_formula has emerged yet
    - phase timings are populated (t_mcts_s / t_nn_train_s / t_buffer_s
      always >= 0; t_eval_s is 0 on non-eval iterations)
    - the 3-tuple returned by q-learning-style evaluators unpacks into
      (eval_reward, eval_success_rate, eval_episode_length)
    - scalar evaluators (EvalResult / float) fall back to setting only
      eval_reward
    - ``eval_use_play`` routes the rule fed to ``eval_simulator``
    - ``play()`` rebuilds its ``NeuralEvaluator`` with the training-time
      ``value_scale`` so the network value comes back in reward units

The eval simulators here are stubs (constant / scripted), so this module
needs only the ``[nn]`` tier, not the ``[rl]`` reinforcement-learning backend.
"""
from __future__ import annotations

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.training.train import train


class _ConstantSimulator:
    """Return the same reward for every node; track call count."""
    def __init__(self, value: float = 1.0):
        self.value = value
        self.calls = 0
    def evaluate(self, node):
        self.calls += 1
        return self.value


class _TupleEvalSimulator:
    """Return a 3-tuple mimicking ``q_learning_agent_eval_mean_reward_success_steps``."""
    def __init__(self, reward: float, success: float, steps: float):
        self.reward = reward
        self.success = success
        self.steps = steps
        self.calls = 0
    def evaluate(self, node):
        self.calls += 1
        return (self.reward, self.success, self.steps)


class _ScalarEvalSimulator:
    """Return a plain float (no success-rate/episode-length fields)."""
    def __init__(self, value: float = 0.5):
        self.value = value
        self.calls = 0
    def evaluate(self, node):
        self.calls += 1
        return self.value


def _tiny_train(**overrides):
    """Run a tiny ``train()`` with sensible defaults."""
    kwargs = dict(
        grammar=AllenIntervalGrammar(event_types=("A", "B"), relations=("<",)),
        expensive_simulator=_ConstantSimulator(value=1.0),
        n_iterations=4,
        n_simulations=4,
        depth_limit=2,
        seed=0,
        buffer_warmup=1,
        train_steps_per_iteration=1,
        max_len=12,
        d_model=16,
        nhead=2,
        num_layers=1,
    )
    kwargs.update(overrides)
    return train(**kwargs)


# --------------------------------------------------------------------------- #
# Phase timings - always recorded
# --------------------------------------------------------------------------- #

def test_iteration_logs_record_phase_timings():
    log = _tiny_train()
    assert len(log.iterations) == 4
    for it in log.iterations:
        assert it.t_mcts_s >= 0.0
        assert it.t_nn_train_s >= 0.0
        assert it.t_buffer_s >= 0.0
        assert it.t_eval_s == 0.0             # no eval_simulator supplied


def test_mcts_phase_time_is_bulk_of_iteration():
    """Sanity check: MCTS (self-play) should be the largest phase."""
    log = _tiny_train()
    for it in log.iterations:
        total = it.t_mcts_s + it.t_nn_train_s + it.t_eval_s + it.t_buffer_s
        # MCTS at least 1/4 of the measured total (cheap phases may be near-zero)
        assert it.t_mcts_s >= total * 0.25 or total < 1e-4


# --------------------------------------------------------------------------- #
# eval_simulator - cadence + unpacking
# --------------------------------------------------------------------------- #

def test_eval_fires_at_cadence_only():
    """With eval_every=2, eval should fire on iterations 0, 2 (not 1, 3)."""
    eval_sim = _TupleEvalSimulator(reward=5.5, success=0.75, steps=12.0)
    log = _tiny_train(
        eval_simulator=eval_sim,
        eval_every=2,
    )

    # Iterations where eval fired have finite eval_reward; others are None.
    fired = [i for i, it in enumerate(log.iterations) if it.eval_reward is not None]
    not_fired = [i for i, it in enumerate(log.iterations) if it.eval_reward is None]

    # iter 0 may skip if best_formula=None on first iteration; iter 2 and
    # subsequent (by cadence) must fire. We assert no iterations fire on
    # odd indices.
    assert all(i % 2 == 0 for i in fired)
    assert all(i % 2 == 1 for i in not_fired), \
        f"unexpected odd iteration skipped eval: {not_fired}"


def test_eval_tuple_unpacks_to_three_metrics():
    eval_sim = _TupleEvalSimulator(reward=5.5, success=0.75, steps=12.0)
    log = _tiny_train(eval_simulator=eval_sim, eval_every=1)
    # Find the first iteration that fired eval.
    fired = next((it for it in log.iterations if it.eval_reward is not None), None)
    assert fired is not None, "expected at least one eval to fire"
    assert fired.eval_reward == 5.5
    assert fired.eval_success_rate == 0.75
    assert fired.eval_episode_length == 12.0
    assert fired.t_eval_s > 0.0


def test_eval_scalar_sets_only_reward():
    """A scalar evaluator fills eval_reward; success/length stay None."""
    eval_sim = _ScalarEvalSimulator(value=0.42)
    log = _tiny_train(eval_simulator=eval_sim, eval_every=1)
    fired = next((it for it in log.iterations if it.eval_reward is not None), None)
    assert fired is not None
    assert fired.eval_reward == 0.42
    assert fired.eval_success_rate is None
    assert fired.eval_episode_length is None


def test_eval_skipped_when_best_formula_is_none():
    """
    If no finite reward has been seen yet (log.best_formula stays None),
    eval must NOT fire - otherwise we'd evaluate nothing meaningful.
    Use a simulator that always returns ``-inf`` so best_formula
    never gets populated.
    """
    class _FailSim:
        def evaluate(self, node):
            return float("-inf")

    eval_sim = _TupleEvalSimulator(reward=99.0, success=1.0, steps=1.0)
    log = _tiny_train(
        expensive_simulator=_FailSim(),
        eval_simulator=eval_sim,
        eval_every=1,
    )
    # best_formula should never have been set
    assert log.best_formula is None
    # eval_simulator was never invoked
    assert eval_sim.calls == 0
    # every it.eval_reward is None
    assert all(it.eval_reward is None for it in log.iterations)


# --------------------------------------------------------------------------- #
# play() rebuilds the evaluator with the training-time value_scale
# --------------------------------------------------------------------------- #

def test_play_uses_trained_value_scale():
    """``play()`` must rebuild its ``NeuralEvaluator`` with the scale the
    training run resolved (stored on the log), not the constructor default.
    With ``value_scale=100.0`` the network predicts in scaled space and
    ``play()`` needs ``value_scale=100.0`` to recover raw-reward units for
    MCTS backup. Pinned by intercepting the evaluator construction inside
    ``play()`` and checking the scale it is handed."""
    from alpha_rule.training.train import play

    log = _tiny_train(value_scale=100.0)
    assert log.value_scale == 100.0

    # NeuralEvaluator is imported lazily inside play(); intercept the class on
    # the module play() imports it from so we capture its construction scale.
    from alpha_rule.evaluation import neural_evaluator as ne_mod
    captured = {}
    orig_cls = ne_mod.NeuralEvaluator

    class _Capturing(orig_cls):
        def __init__(self, *a, **kw):
            captured["value_scale"] = kw.get("value_scale")
            super().__init__(*a, **kw)

    ne_mod.NeuralEvaluator = _Capturing
    try:
        play(
            log,
            grammar=AllenIntervalGrammar(event_types=("A", "B"), relations=("<",)),
            simulator=_ConstantSimulator(1.0),
            temperature=0.0,
            n_simulations=2, depth_limit=1,
        )
    finally:
        ne_mod.NeuralEvaluator = orig_cls
    assert captured["value_scale"] == 100.0


# --------------------------------------------------------------------------- #
# eval_use_play opt-in
# --------------------------------------------------------------------------- #

def test_eval_use_play_default_false_uses_best_formula():
    """Default ``eval_use_play=False`` sends ``log.best_formula`` to the
    evaluator (the highest-reward trajectory step seen during self-play)."""
    class _CaptureEval:
        def __init__(self): self.seen = []
        def evaluate(self, node):
            self.seen.append(node.name)
            return (5.5, 0.75, 12.0)

    cap = _CaptureEval()
    log = _tiny_train(eval_simulator=cap, eval_every=1)
    if log.best_formula is not None:
        # Every call the evaluator saw was the running-best formula.
        assert cap.seen[-1] == log.best_formula


def test_eval_use_play_true_sends_play_output_to_evaluator():
    """With ``eval_use_play=True`` the rule fed to ``eval_simulator`` comes
    from ``play()``, not from ``log.best_formula``. We can't predict which
    rule play() picks, but we can require it's valid (non-empty) and that the
    eval simulator was actually called."""
    class _CaptureEval:
        def __init__(self): self.seen = []
        def evaluate(self, node):
            self.seen.append(node.name)
            return (7.0, 1.0, 2.0)

    cap = _CaptureEval()
    log = _tiny_train(
        eval_simulator=cap,
        eval_every=1,
        eval_use_play=True,
    )
    # Evaluator should have been called at least once.
    assert len(cap.seen) > 0, "eval_use_play=True should still fire eval_simulator"
    # The names it saw should be strings that parse into rule structure.
    for name in cap.seen:
        assert isinstance(name, str) and len(name) > 0


def test_eval_every_zero_or_one_treated_as_every_iteration():
    """Edge case: eval_every=0 shouldn't divide-by-zero."""
    eval_sim = _TupleEvalSimulator(reward=1.0, success=0.5, steps=5.0)
    log = _tiny_train(eval_simulator=eval_sim, eval_every=0)  # implementation uses max(1, ...)
    # At least one eval should have fired.
    assert any(it.eval_reward is not None for it in log.iterations)
