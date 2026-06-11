"""
Outer training loop: self-play, replay, NN update, repeat.

This is the MVP entry-point that ties together everything the other components
built. It is intentionally small, the moving parts live in their own
modules, and this file just wires them.

Usage::

    from alpha_rule.evaluation.rule_simulator import RuleSimulator
    from alpha_rule.grammar.allen import AllenIntervalGrammar
    from alpha_rule.training.train import train

    grammar = AllenIntervalGrammar(event_types=("A","B","C"))
    expensive = RuleSimulator(env_name="OpenTheChests-v0", ...)

    log = train(
        grammar=grammar,
        expensive_simulator=expensive,
        n_iterations=100,
        n_simulations=50,
        depth_limit=5,
    )

Returns a ``TrainingLog`` with the per-iteration loss values and the
best-formula seen so far.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Set, Tuple

import numpy as np

from alpha_rule.evaluation.evaluator import Evaluator, RuleStringNode
from alpha_rule.grammar.grammar import Grammar
from alpha_rule.mcts.backprop import (
    BackpropStrategy,
    MaxRewardBackup,
    PercentileRewardBackup,
)
from alpha_rule.mcts.replay import ReplayBuffer, Trajectory
from alpha_rule.mcts.selection import PUCTSelection, SelectionStrategy
from alpha_rule.mcts.self_play import run_self_play


@dataclass
class IterationLog:
    """One row of the training log: everything measured for a single
    self-play + train iteration. The ``AlphaZeroCSVLogger`` callback turns
    one of these into one CSV row."""

    iteration: int
    """Zero-based iteration index (the loop counter in ``train``)."""

    trajectory_length: int
    """Number of construction steps the self-play episode produced, i.e.
    ``len(traj.steps)`` from ``run_self_play``. Bounded by ``depth_limit``;
    shorter when the episode hit a terminal rule or a dead subtree early."""

    best_reward_in_trajectory: float
    """Highest finite simulator reward seen among this episode's steps, as
    picked by ``_best_in_trajectory``. ``-inf`` if every step failed."""

    n_failed_evaluations: int
    """How many of this episode's chosen steps scored a non-finite reward
    (``-inf``, a structural match failure), counted by ``_failed_count`` on
    the raw trajectory before any dead-rule bookkeeping."""

    train_total: float = 0.0
    """Mean total loss across this iteration's ``train_step`` calls
    (policy + value, each weighted). ``0.0`` while the buffer is still
    below ``buffer_warmup`` and no training ran."""

    train_policy: float = 0.0
    """Mean policy (soft-label cross-entropy) loss from ``train_step``."""

    train_value: float = 0.0
    """Mean value (MSE) loss from ``train_step``."""

    n_dead_rules: int = 0
    """Cumulative number of distinct rule names known to score ``-inf`` so
    far (the size of ``train``'s persistent ``dead_rules`` set after this
    iteration). Grows monotonically; how much of the search space has been
    pruned out of future simulator calls."""

    buffer_fill_fraction: float = 0.0
    """Fraction of the replay buffer's capacity in use after this
    iteration's push, in ``[0, 1]`` (``ReplayBuffer.fill_fraction``). Near
    1.0 means the buffer is evicting old targets each iteration; very low
    means the capacity dwarfs your throughput."""

    best_formula_in_trajectory: Optional[str] = None
    """Name of the rule that earned ``best_reward_in_trajectory`` this
    iteration. ``None`` when the trajectory had no finite-reward steps
    (every evaluation was ``-inf``). The ``AlphaZeroCSVLogger`` callback
    reads this to populate ``running_best_formula``."""

    # --- Eval metrics (populated when eval_simulator fires this iter) --- #
    eval_reward: Optional[float] = None
    """Mean episodic return of the running-best formula in the
    exploitation env. ``None`` on iterations where the eval didn't fire
    (gated by ``eval_every``) or when no best_formula exists yet."""
    eval_success_rate: Optional[float] = None
    """Fraction of eval episodes with total reward >= 1."""
    eval_episode_length: Optional[float] = None
    """Mean episode length (steps) across eval episodes."""

    # --- Per-iteration phase timings (seconds) ---------------------- #
    t_mcts_s: float = 0.0
    """Wall time spent inside ``run_self_play`` this iteration."""
    t_nn_train_s: float = 0.0
    """Wall time spent in ``train_step`` calls this iteration."""
    t_eval_s: float = 0.0
    """Wall time spent inside ``eval_simulator.evaluate`` this iteration
    (``0.0`` on iterations where the eval didn't fire)."""
    t_buffer_s: float = 0.0
    """Wall time spent in ``buffer.push_trajectory`` this iteration."""


@dataclass
class TrainingLog:
    """Return value of ``train``: the per-iteration history plus the
    running-best rule and the artefacts ``play`` needs to run the trained
    policy."""

    iterations: List[IterationLog] = field(default_factory=list)
    """One ``IterationLog`` per iteration, in order."""

    best_formula: Optional[str] = None
    """Name of the highest-reward rule seen across all self-play so far
    (tie-broken toward the longer rule by ``_best_in_trajectory``).
    Exploration logbook, not the trained policy's own answer -- use
    ``play`` for that."""

    best_reward: float = float("-inf")
    """Simulator reward of ``best_formula``; ``-inf`` until a finite-reward
    rule is found."""

    device: Optional[str] = None
    """Resolved torch device for the policy-value network, e.g. ``"cuda:0"``
    or ``"cpu"``. Set once at the start of ``train()``."""

    # --- Artefacts needed for AlphaZero-style inference via ``play()`` --- #
    model: Optional[Any] = None
    """Trained ``AllenFormulaNet``. Populated at the end of ``train()`` so
    ``play(log, ...)`` can run a greedy rollout without reconstructing the
    network. ``None`` on a bare ``TrainingLog()``."""
    max_len: Optional[int] = None
    """Token budget used to build the model / ``NeuralEvaluator``. Needed
    by ``play()`` to reconstruct a ``NeuralEvaluator`` around ``model``."""
    n_simulations: Optional[int] = None
    """MCTS simulations per step used during training. ``play()`` reuses
    this as its default search budget."""
    depth_limit: Optional[int] = None
    """Construction-step budget used during training. ``play()`` reuses
    this as its default rollout depth."""
    value_scale: Optional[float] = None
    """Positive reward cap used to scale value targets into ``[-1, +1]`` during
    training (the simulator's ``reward_scale``, ``None`` if unknown). ``play()``
    reuses it to build a matching ``NeuralEvaluator`` so MCTS backup sees the
    network value in the same units as the simulator rewards."""


def _failed_count(traj: Trajectory) -> int:
    return sum(1 for s in traj.steps if not math.isfinite(s.reward))


def _collect_dead_rules(traj: Trajectory) -> List[str]:
    """Return rule-names from steps whose reward was non-finite (-inf).

    The returned names describe the CHILD the reward actually rates
    (``step.next_state``), falling back to ``step.state`` for rows that
    didn't populate ``next_state``.
    """
    out: List[str] = []
    for step in traj.steps:
        if math.isfinite(step.reward):
            continue
        candidate = step.next_state if step.next_state is not None else step.state
        name = (
            candidate if isinstance(candidate, str)
            else getattr(candidate, "name", None)
        )
        if name:
            out.append(name)
    return out


def _best_in_trajectory(traj: Trajectory) -> tuple:
    """
    Pick the (state, reward) pair describing the best rule encountered.

    Prefers ``step.next_state`` (the child the reward actually
    describes) over ``step.state`` (the parent where MCTS was rooted).
    Without this preference the "best formula" would surface the
    parent state, typically ``"<ROOT>"``, whenever the first action
    happened to be the best-rewarded child.

    Tie-break: on equal finite reward, the **longer** rule wins
    (token count via ``name.split()``). The ``-|distance|`` reward
    saturates at 0, so length-1 rules frequently tie with longer,
    more-specific rules, preferring the longer formula pushes the
    search toward more expressive candidates instead of getting stuck
    on the first single-event placeholder.
    """
    import math
    best_reward = float("-inf")
    best_state = None
    best_len = -1
    for step in traj.steps:
        if not math.isfinite(step.reward):
            continue
        candidate = step.next_state if step.next_state is not None else step.state
        candidate_name = candidate.name if hasattr(candidate, "name") else str(candidate)
        candidate_len = len(candidate_name.split())
        strict_better = step.reward > best_reward
        tie_longer = step.reward == best_reward and candidate_len > best_len
        if strict_better or tie_longer:
            best_reward = step.reward
            best_state = candidate
            best_len = candidate_len
    return best_state, best_reward


def _unpack_eval_tuple(raw):
    """
    Turn whatever ``eval_simulator.evaluate(...)`` returned into a
    ``(reward, success_rate, episode_length)`` triple.

    ``q_learning_agent_eval_mean_reward_success_steps`` returns a
    3-tuple of (mean_reward, success_rate, mean_steps). ``RuleSimulator``
    will usually return that tuple unchanged. A caller that swapped in
    a scalar evaluator (e.g., the distance evaluator) only has the
    reward, we record it under ``eval_reward`` and leave the other two
    as ``None``.
    """
    # EvalResult has a ``.value`` attribute; fall through to the scalar path.
    value_attr = getattr(raw, "value", None)
    if value_attr is not None and not isinstance(raw, tuple):
        return float(value_attr), None, None
    if isinstance(raw, tuple) and len(raw) >= 3:
        return float(raw[0]), float(raw[1]), float(raw[2])
    if isinstance(raw, tuple) and len(raw) >= 1:
        return float(raw[0]), None, None
    return float(raw), None, None


def _warn_risky_config(
    *,
    max_len: int,
    depth_limit: int,
    explicit_value_scale: Optional[float],
    simulator: Evaluator,
    backup,
    selection,
    q_source: str,
    eval_simulator,
    eval_every: int,
) -> None:
    """Emit a ``warnings.warn`` for each known footgun in a ``train`` config.

    Covers the combinations that don't fail loudly on their own: a token
    budget too small for the search depth (crashes mid-episode), a value
    scale that disagrees with the simulator's reward scale (trains on a
    mis-scaled value target), a backup/selection pair that optimise
    different statistics, and an eval cadence that silently runs every
    iteration. All are non-fatal -- the run still starts.
    """
    # 1. Token budget vs search depth. A depth-d rule is d productions =
    #    d + 2 token ids (BOS + EOS). The MCTS search is capped at depth_limit
    #    (see _run_one_round), so the deepest node ever encoded sits at exactly
    #    depth_limit and max_len >= depth_limit + 2 is both necessary AND
    #    sufficient; below it, encode() raises inside the search.
    if max_len < depth_limit + 2:
        warnings.warn(
            f"max_len={max_len} < depth_limit + 2 = {depth_limit + 2}: a rule "
            f"built to depth_limit needs depth_limit + 2 token ids (productions "
            f"+ BOS + EOS). The deepest self-play episodes will overflow the "
            f"tokenizer mid-search and raise. Raise max_len to at least "
            f"{depth_limit + 2}.",
            stacklevel=3,
        )

    # 2. Explicit value_scale that disagrees with the simulator's reward cap.
    #    The network value (multiplied by value_scale) and the simulator's raw
    #    rewards (capped at reward_scale) then live on different scales in the
    #    shared MCTS backup, and the clip to [-1, 1] is wrong for one of them.
    sim_reward_scale = getattr(simulator, "reward_scale", None)
    if (
        explicit_value_scale is not None
        and sim_reward_scale is not None
        and abs(float(explicit_value_scale) - float(sim_reward_scale)) > 1e-9
    ):
        warnings.warn(
            f"value_scale={explicit_value_scale} differs from the simulator's "
            f"reward_scale={sim_reward_scale}: the network value and the "
            f"simulator's raw rewards then sit on different scales in the shared "
            f"MCTS backup, and the value target's clip to [-1, 1] is wrong for "
            f"one of them. Pass value_scale=None to inherit reward_scale, or "
            f"match them on purpose.",
            stacklevel=3,
        )

    # 3. Percentile backup needs PUCT to read the percentile-filtered mean.
    #    q_source is only consulted when selection is built from the scalars
    #    (selection is None); an explicit selection object owns its own q_source.
    if (
        isinstance(backup, str) and backup == "percentile"
        and selection is None and q_source != "filtered_mean"
    ):
        warnings.warn(
            f"backup='percentile' pairs with q_source='filtered_mean', but "
            f"q_source={q_source!r}: PUCT would read Q_max while the backup fills "
            f"the percentile-filtered mean, so selection and backup optimise "
            f"different statistics. Set q_source='filtered_mean'.",
            stacklevel=3,
        )

    # 4. eval cadence. The gate uses `it % max(1, eval_every)`, so a
    #    non-positive eval_every silently runs the (expensive) eval EVERY
    #    iteration instead of disabling it.
    if eval_simulator is not None and eval_every <= 0:
        warnings.warn(
            f"eval_every={eval_every} <= 0: the eval gate floors it to 1, so the "
            f"(expensive) eval runs on every iteration. Use a positive cadence, "
            f"or pass eval_simulator=None to turn eval off.",
            stacklevel=3,
        )


def train(
    *,
    grammar: Grammar,
    expensive_simulator: Evaluator,
    # --- search ---------------------------------------------------- #
    n_iterations: int = 100,
    n_simulations: int = 50,
    depth_limit: int = 5,
    temperature: float = 1.0,
    temperature_final: Optional[float] = 0.1,
    # --- replay buffer --------------------------------------------- #
    buffer_capacity: int = 10_000,
    buffer_warmup: int = 16,
    batch_size: int = 16,
    value_scale: Optional[float] = None,
    # --- network (AllenFormulaNet) --------------------------------- #
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.0,
    max_len: int = 64,
    # --- optimisation ---------------------------------------------- #
    train_steps_per_iteration: int = 4,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 0.0,
    value_weight: float = 1.0,
    policy_weight: float = 1.0,
    # --- selection (PUCT) ------------------------------------------ #
    c_puct: float = 1.5,
    fpu_reduction: float = 0.25,
    q_source: str = "max",
    selection: Optional[SelectionStrategy] = None,
    # --- backpropagation ------------------------------------------- #
    backup: str | BackpropStrategy = "max",
    percentile: float = 20.0,
    min_samples: int = 10,
    # --- exploration / leaf evaluation ----------------------------- #
    dirichlet_eps: float = 0.25,
    dirichlet_alpha: float = 0.3,
    leaf_eval_mode: str = "nn",
    n_chosen_evals: int = 1,
    # --- evaluation / logging / misc ------------------------------- #
    eval_simulator: Optional[Evaluator] = None,
    eval_every: int = 5,
    eval_use_play: bool = False,
    on_iteration_end=None,
    device: Optional[str] = None,
    seed: int = 0,
) -> TrainingLog:
    """
    Run ``n_iterations`` self-play + train iterations.

    Args:
        grammar: production set.
        expensive_simulator: ``Evaluator`` (e.g. ``RuleSimulator``) used
            for every leaf evaluation in MCTS and for the reward of the
            chosen production. The MVP doesn't separate "cheap" vs
            "expensive" evaluators yet, that's the deferred two-tier
            work.
        n_iterations: number of self-play episodes.
        n_simulations: MCTS simulations per construction step.
        depth_limit: max construction steps per self-play episode.
        temperature: starting temperature for the visit-count ACTION
            sampling at iteration 0. The stored policy target is always the
            tau=1 visit distribution (decoupled inside ``run_self_play``), so
            annealing the sampling temperature sharpens which rule gets
            committed without distorting the training signal.
        temperature_final: temperature reached at the last iteration. The
            sampling temperature is linearly annealed from ``temperature``
            (iteration 0) to ``temperature_final`` (iteration
            ``n_iterations - 1``): high early to explore, low late to commit
            to what the trained net prefers. ``None`` (or equal to
            ``temperature``) keeps it constant. Default ``0.1``.
        buffer_capacity, buffer_warmup, batch_size,
        train_steps_per_iteration: replay-buffer + training cadence.
        d_model, nhead, num_layers, max_len: ``AllenFormulaNet`` config.
        learning_rate: Adam lr.
        seed: master seed for numpy / torch RNGs.
        eval_simulator: optional second ``Evaluator`` used to measure
            how well the **running-best formula so far** performs in
            the "exploitation" env. Typically a ``RuleSimulator`` with
            ``q_learning_agent_eval_mean_reward_success_steps`` as the
            ``agent_eval``; returns ``(mean_reward, success_rate,
            mean_steps)``. Each call is expensive (trains a fresh
            Q-learning agent), so the cadence is controlled by
            ``eval_every``.
        eval_every: run ``eval_simulator`` on iterations where
            ``iteration % eval_every == 0`` (default 5). Ignored when
            ``eval_simulator is None``.
        on_iteration_end: optional callback ``fn(log: IterationLog)``.
        device: torch device for the policy-value network, e.g.
            ``"cuda"``, ``"cuda:0"``, ``"cpu"``. Defaults to ``"cuda"``
            when ``torch.cuda.is_available()`` else ``"cpu"``. The
            replay-buffer, MCTS, and expensive-simulator side stay on
            the host; only the NN forward/backward runs on ``device``.
        n_chosen_evals: number of independent simulator samples
            averaged when evaluating the chosen node at each
            construction step (default 1). Raising this reduces
            Q-learning eval variance. See ``run_self_play`` for details.
        backup: backprop strategy, the string ``"max"`` (default,
            ``MaxRewardBackup``) or ``"percentile"`` (``PercentileRewardBackup``
            built from ``percentile`` / ``min_samples``). You may also pass
            an explicit ``BackpropStrategy`` object (advanced; takes
            precedence over the string form).
        selection: optional ``SelectionStrategy`` object. ``None``
            (default) builds ``PUCTSelection(c_puct, fpu_reduction,
            q_source)`` from the scalar kwargs; passing an object takes
            precedence (the scalar PUCT kwargs are then ignored). With
            ``backup="percentile"`` set ``q_source="filtered_mean"`` so
            PUCT reads the percentile-filtered mean instead of ``Q_max``.
        value_scale: positive reward cap used to scale value targets into
            ``[-1, +1]`` in the replay buffer and, matching, to scale the
            network's value back to reward units in the ``NeuralEvaluator``.
            ``None`` (default) reads the cap from the simulator's
            ``reward_scale`` if it has one, else falls back to ``1.0``.
        leaf_eval_mode: ``"nn"`` (default, AlphaZero leaf bootstrap)
            uses the network's value head at non-terminal leaves and
            the simulator only at terminal leaves; ``"simulator"`` uses
            the simulator at every leaf. The chosen-step reward written
            to the trajectory is always from the simulator regardless of
            this flag.
        grad_clip: if > 0, clip the global gradient L2-norm during
            every ``train_step`` before ``optimizer.step()``. Default
            ``0.0`` disables clipping (the default). A value
            like ``1.0`` stabilises Adam's second-moment estimate when
            targets are large or heavy-tailed.
        dirichlet_eps: AlphaZero-style Dirichlet noise weight applied
            to the MCTS search root's children's priors at each
            construction step. ``p' = (1-eps) * p + eps * Dir(alpha)``.
            Default ``0.25`` (on): it forces residual exploration of
            branches PUCT would otherwise permanently starve (e.g. branches
            whose first expansion returned a very negative Q), directly
            addressing the "buffer has 0 BAD rows" finding from
            ``accuracy_convergence.ipynb``. Set ``0.0`` to disable, which
            also makes a seeded run bit-identical across executions (the
            noise is the only non-reproducible-by-config element).
        dirichlet_alpha: Dirichlet concentration. Lower = spikier
            noise, higher = more uniform. Ignored when
            ``dirichlet_eps == 0``. Default ``0.3`` matches the
            AlphaZero paper's Go value; small games or small vocabs
            may prefer ~``1.0``.
        eval_use_play: when True, the periodic ``eval_simulator``
            snapshot evaluates the rule produced by a fresh ``play()``
            MCTS rollout (the trained policy's own answer) rather
            than ``log.best_formula`` (the highest-reward trajectory
            step seen during self-play). Default ``False`` preserves
            the default log contents; ``True`` is the AlphaZero-standard
            "evaluate the learned agent" recipe. NOTE: ``play()``
            re-runs a full MCTS search per eval, so turning this on
            roughly doubles per-eval wall time, combine with a
            larger ``eval_every`` if that matters.
        dim_feedforward: Transformer feed-forward width (default 128).
        dropout: Transformer dropout probability (default 0.0).
        weight_decay: Adam L2 weight-decay coefficient (default 0.0).
        value_weight, policy_weight: relative weights of the value (MSE)
            and policy (cross-entropy) loss terms in ``train_step``
            (default 1.0 each).
        c_puct, fpu_reduction, q_source: ``PUCTSelection`` knobs used to
            build the default selection strategy (see ``selection``).
            ``q_source`` is ``"max"`` (read ``Q_max``) or
            ``"filtered_mean"`` (read the percentile-filtered mean).
            NOTE on ``c_puct`` vs reward magnitude: PUCT scores each child
            as ``Q + c_puct * P * sqrt(sum_N) / (1 + N)`` where ``Q`` is the
            node's value in RAW reward units (capped at ``value_scale`` /
            the simulator's ``reward_scale``, typically ~1-4), not
            normalised to ``[-1, 1]``. So ``c_puct`` is implicitly tied to
            your reward magnitude: the exploration term must stay comparable
            to ``Q`` to matter. The default ``1.5`` suits rewards capped
            near 1; if your ``reward_scale`` is larger (say 4), scale
            ``c_puct`` up roughly in proportion or the search collapses to
            pure exploitation.
        percentile, min_samples: ``PercentileRewardBackup`` knobs, used
            only when ``backup="percentile"``.

    Returns:
        ``TrainingLog`` with per-iteration metrics and the best
        formula encountered.
    """
    # Lazy heavy imports so test discovery doesn't pay the torch cost.
    import torch
    from alpha_rule.evaluation.neural_evaluator import NeuralEvaluator
    from alpha_rule.nn.model import AllenFormulaNet
    from alpha_rule.nn.tokenizer import GrammarTokenizer
    from alpha_rule.nn.training import train_step

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Resolve the compute device. Default: CUDA when available, else CPU.
    # Tensors built inside ``train_step`` / ``NeuralEvaluator`` are moved
    # to the model's device at call time, so passing e.g. ``"cuda:0"``
    # here is sufficient.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but torch.cuda.is_available() is False. "
            f"This usually means torch was installed as the CPU-only build "
            f"({torch.__version__}). Install a CUDA build, e.g. "
            f"pip install --index-url https://download.pytorch.org/whl/cu121 "
            f"torch --upgrade"
        )

    tokenizer = GrammarTokenizer(grammar)
    model = AllenFormulaNet(
        tokenizer,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_len=max_len,
        dropout=dropout,
    ).to(torch_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    # Wire the evaluator's value scale to the simulator's reward cap (an explicit
    # value_scale wins) so the network value and the simulator reward share one
    # scale in the MCTS backup.
    if value_scale is not None:
        network_evaluator = NeuralEvaluator(
            model, grammar, max_len=max_len, value_scale=value_scale,
        )
    else:
        network_evaluator = NeuralEvaluator.from_simulator(
            model, grammar, expensive_simulator, max_len=max_len,
        )
    resolved_scale = network_evaluator.value_scale
    buffer = ReplayBuffer(capacity=buffer_capacity, value_scale=resolved_scale)

    # Resolve the search strategies once. ``selection``/``backup`` accept an
    # explicit object (advanced override, takes precedence); otherwise they
    # are built from the scalar kwargs so every knob is visible at this call.
    sel = selection if selection is not None else PUCTSelection(
        c_puct=c_puct, fpu_reduction=fpu_reduction, q_source=q_source,
    )
    if isinstance(backup, str):
        if backup == "max":
            bp: BackpropStrategy = MaxRewardBackup()
        elif backup == "percentile":
            bp = PercentileRewardBackup(percentile=percentile, min_samples=min_samples)
        else:
            raise ValueError(
                f"backup must be 'max', 'percentile', or a BackpropStrategy "
                f"object; got {backup!r}"
            )
    else:
        bp = backup

    # --- Risky / inconsistent configuration warnings ----------------- #
    # Non-fatal: surface combinations that silently misbehave so a long run
    # doesn't crash mid-search or train against a mis-scaled signal.
    _warn_risky_config(
        max_len=max_len,
        depth_limit=depth_limit,
        explicit_value_scale=value_scale,
        simulator=expensive_simulator,
        backup=backup,
        selection=selection,
        q_source=q_source,
        eval_simulator=eval_simulator,
        eval_every=eval_every,
    )

    log = TrainingLog(
        device=str(torch_device),
        max_len=max_len,
        n_simulations=n_simulations,
        depth_limit=depth_limit,
        value_scale=resolved_scale,
    )
    # Expose the in-training model immediately so ``play()`` can be called
    # from the eval hook (``eval_use_play=True``) while training is still
    # running. Python references are sticky, the same object gets trained
    # in place, so ``log.model`` stays current at every iteration.
    log.model = model

    # Persistent set of rule names already known to evaluate to -inf.
    # Accumulated across self-play episodes and passed into run_self_play
    # so MCTS expansion can mark matching children dead before any rollout
    # reaches them, saves a full simulator.evaluate (= Q-learning training
    # run) per revisit.
    dead_rules: Set[str] = set()

    for it in range(n_iterations):
        # Linearly anneal the action-sampling temperature from `temperature`
        # (iteration 0) to `temperature_final` (last iteration). The policy
        # TARGET stored in the trajectory stays at tau=1 inside run_self_play,
        # so this only sharpens which rule gets committed, not the training
        # signal. `temperature_final=None` (or == temperature) keeps it fixed.
        if temperature_final is None or n_iterations <= 1:
            temp_it = temperature
        else:
            frac = it / (n_iterations - 1)
            temp_it = temperature + (temperature_final - temperature) * frac

        # --- MCTS / self-play -------------------------------------- #
        t0 = time.perf_counter()
        traj = run_self_play(
            grammar=grammar,
            simulator=expensive_simulator,
            network_evaluator=network_evaluator,
            n_simulations=n_simulations,
            depth_limit=depth_limit,
            temperature=temp_it,
            selection=sel,
            backup=bp,
            rng=rng,
            n_chosen_evals=n_chosen_evals,
            dirichlet_eps=dirichlet_eps,
            dirichlet_alpha=dirichlet_alpha,
            leaf_eval_mode=leaf_eval_mode,
            value_scale=resolved_scale,
            dead_rule_names=dead_rules if dead_rules else None,
        )
        t_mcts_s = time.perf_counter() - t0

        # Count this episode's failed (-inf) evaluations from the raw
        # trajectory BEFORE folding those rules into the persistent dead set,
        # so the metric reflects what this episode actually saw independent of
        # the dead-rule bookkeeping below.
        n_failed = _failed_count(traj)

        # Accumulate -inf rules seen in this episode into the persistent
        # set so future iterations can short-circuit them.
        dead_rules.update(_collect_dead_rules(traj))

        # --- Replay buffer push ------------------------------------ #
        t0 = time.perf_counter()
        buffer.push_trajectory(traj)
        t_buffer_s = time.perf_counter() - t0

        # --- NN training step ------------------------------------- #
        t0 = time.perf_counter()
        train_total = train_policy = train_value = 0.0
        if len(buffer) >= buffer_warmup:
            for _ in range(train_steps_per_iteration):
                batch = buffer.sample(batch_size)
                step_log = train_step(
                    model, optimizer, batch,
                    max_len=max_len,
                    value_weight=value_weight,
                    policy_weight=policy_weight,
                    grad_clip=grad_clip,
                )
                train_total += step_log.total
                train_policy += step_log.policy
                train_value += step_log.value
            train_total /= max(1, train_steps_per_iteration)
            train_policy /= max(1, train_steps_per_iteration)
            train_value /= max(1, train_steps_per_iteration)
        t_nn_train_s = time.perf_counter() - t0

        # --- Running best (updated BEFORE the eval so eval sees it) - #
        best_state, best_reward = _best_in_trajectory(traj)
        best_state_str = (
            best_state if isinstance(best_state, str)
            else getattr(best_state, "name", None) if best_state is not None
            else None
        )
        if best_state is not None:
            candidate_name = best_state_str or str(best_state)
            candidate_len = len(candidate_name.split())
            prev_len = len(log.best_formula.split()) if log.best_formula else -1
            strict_better = best_reward > log.best_reward
            tie_longer = best_reward == log.best_reward and candidate_len > prev_len
            if strict_better or tie_longer:
                log.best_formula = candidate_name
                log.best_reward = best_reward

        # --- Optional eval ---------------------------------------- #
        # Two modes:
        #   - ``eval_use_play=False`` (default): evaluate ``log.best_formula``,
        #     the highest-reward trajectory step seen during self-play. This
        #     is the default path; fast but leaks "exploration luck" into
        #     the reported eval number.
        #   - ``eval_use_play=True``: run ``play()`` to get the trained
        #     policy's own committed rule (argmax-MCTS with temperature=0
        #     using the current network for priors + value bootstrap), then
        #     evaluate that. This is the AlphaZero-standard "evaluate the
        #     learned agent" recipe. Costs ~one extra self-play round per
        #     eval, so combine with a larger ``eval_every`` in long runs.
        t_eval_s = 0.0
        eval_reward = eval_success = eval_steps = None
        if eval_simulator is not None and it % max(1, eval_every) == 0:
            t0 = time.perf_counter()
            rule_to_eval: Optional[str] = None
            if eval_use_play:
                try:
                    play_rule, _play_reward = play(
                        log,
                        grammar=grammar,
                        simulator=expensive_simulator,
                        temperature=0.0,
                        selection=sel,
                        backup=bp,
                    )
                    rule_to_eval = play_rule
                except Exception:
                    # Defensive: if play() fails (e.g. empty buffer edge
                    # cases on the first iterations) fall back to the
                    # default path rather than bringing down training.
                    rule_to_eval = log.best_formula
            else:
                rule_to_eval = log.best_formula
            if rule_to_eval is not None:
                raw = eval_simulator.evaluate(RuleStringNode(name=rule_to_eval))
                eval_reward, eval_success, eval_steps = _unpack_eval_tuple(raw)
            t_eval_s = time.perf_counter() - t0

        it_log = IterationLog(
            iteration=it,
            trajectory_length=len(traj.steps),
            best_reward_in_trajectory=best_reward,
            n_failed_evaluations=n_failed,
            train_total=train_total,
            train_policy=train_policy,
            train_value=train_value,
            n_dead_rules=len(dead_rules),
            buffer_fill_fraction=buffer.fill_fraction,
            best_formula_in_trajectory=best_state_str,
            eval_reward=eval_reward,
            eval_success_rate=eval_success,
            eval_episode_length=eval_steps,
            t_mcts_s=t_mcts_s,
            t_nn_train_s=t_nn_train_s,
            t_eval_s=t_eval_s,
            t_buffer_s=t_buffer_s,
        )
        log.iterations.append(it_log)
        if on_iteration_end is not None:
            on_iteration_end(it_log)

    # ``log.model`` is already populated at the top of train() so the
    # eval hook (and the caller) see the trained model live.
    return log


def play(
    log: TrainingLog,
    *,
    grammar: Grammar,
    simulator: Evaluator,
    net: Optional[Any] = None,
    temperature: float = 0.0,
    n_simulations: Optional[int] = None,
    depth_limit: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    forbidden_root_actions: Optional[Iterable[str]] = None,
    selection: Optional[SelectionStrategy] = None,
    backup: Optional[BackpropStrategy] = None,
) -> Tuple[Optional[str], float]:
    """
    AlphaZero-style greedy rollout using the trained policy/value network.

    Runs one self-play episode with the trained network supplying PUCT
    priors and ``temperature=0`` (so each step picks the argmax of the
    MCTS visit distribution, the canonical AlphaZero "play" mode).
    Returns the final constructed rule and its simulator reward.

    Compared to ``log.best_formula``:
        - ``log.best_formula`` is the highest-reward state observed
          during *training* self-plays. It's an exploration logbook, 
          useful for debugging, but leaks ground-truth reward into the
          "answer" and ignores the trained policy.
        - ``play()`` is the policy's *own* answer: the greedy output of
          MCTS guided by the trained net. No reward-max post-hoc walk.

    Args:
        log: the ``TrainingLog`` returned by ``train()``. Supplies the
            trained model, ``max_len``, and default search budget.
        grammar: grammar to root the rollout in.
        simulator: ``Evaluator`` used to score each chosen production.
        net: optional network override. Defaults to ``log.model``.
        temperature: visit-count sampling temperature. Default ``0``
            means argmax over MCTS visits at each step (deterministic).
            Pass a small positive value for stochastic play.
        n_simulations: MCTS simulations per construction step. Defaults
            to ``log.n_simulations`` (i.e., the training-time budget).
        depth_limit: number of construction steps. Defaults to
            ``log.depth_limit``.
        rng: optional ``np.random.Generator``. Defaults to a fresh one.
        forbidden_root_actions: optional set of root-action names to
            block. ``run_self_play`` pre-expands the root and marks each
            forbidden child as ``is_dead=True`` so MCTS never visits it.
            Used by ``play_top_k`` to enforce branch-level diversity.
        selection: optional ``SelectionStrategy``. Pass the same
            instance you used in ``train()`` to keep inference-time MCTS
            consistent with training-time MCTS (most relevant when you
            paired a non-default ``backup`` with a non-default
            ``q_source`` at training time).
        backup: optional ``BackpropStrategy``. Same consistency note as
            ``selection``.

    Returns:
        ``(rule_name, reward)``. ``rule_name`` is ``None`` and
        ``reward`` is ``-inf`` only when the rollout produced no steps
        at all (e.g., every root child was already dead).
    """
    import math

    import torch

    from alpha_rule.evaluation.neural_evaluator import NeuralEvaluator

    model = net if net is not None else log.model
    if model is None:
        raise ValueError(
            "play(): no trained network available. Pass `net=` explicitly "
            "or call `play(log, ...)` with a log produced by `train()` "
            "(which populates `log.model`)."
        )
    if log.max_len is None:
        raise ValueError(
            "play(): log.max_len is unset; cannot build NeuralEvaluator. "
            "Use a log produced by the current `train()` implementation."
        )

    sims = n_simulations if n_simulations is not None else (log.n_simulations or 50)
    depth = depth_limit if depth_limit is not None else (log.depth_limit or 5)
    rng = rng or np.random.default_rng()

    # Reuse the scale the training run resolved (stored on the log) so the
    # network value comes back in the simulator's reward units; fall back to the
    # simulator's own reward_scale, then to raw passthrough.
    scale = log.value_scale if log.value_scale is not None else getattr(
        simulator, "reward_scale", None
    )
    if scale is not None:
        network_evaluator = NeuralEvaluator(
            model, grammar, max_len=log.max_len, value_scale=scale,
        )
    else:
        network_evaluator = NeuralEvaluator.from_simulator(
            model, grammar, simulator, max_len=log.max_len,
        )

    with torch.no_grad():
        traj = run_self_play(
            grammar=grammar,
            simulator=simulator,
            network_evaluator=network_evaluator,
            n_simulations=sims,
            depth_limit=depth,
            temperature=temperature,
            selection=selection,
            backup=backup,
            rng=rng,
            forbidden_root_actions=forbidden_root_actions,
        )

    if not traj.steps:
        return None, float("-inf")

    # Prefer the deepest *finite*-reward step: the greedy policy's
    # committed rule. If the very last step happened to expand into a
    # dead branch (reward=-inf), fall back to the last live step so the
    # returned rule is still usable. If *every* step failed, return the
    # terminal step as-is (caller can inspect reward=-inf).
    finite_steps = [s for s in traj.steps if math.isfinite(s.reward)]
    chosen = finite_steps[-1] if finite_steps else traj.steps[-1]

    if chosen.next_state is not None:
        rule_name = (
            chosen.next_state.name if hasattr(chosen.next_state, "name")
            else str(chosen.next_state)
        )
    else:
        rule_name = (
            chosen.state.name if hasattr(chosen.state, "name")
            else str(chosen.state)
        )
    return rule_name, chosen.reward


def play_top_k(
    log: TrainingLog,
    *,
    grammar: Grammar,
    simulator: Evaluator,
    k: int,
    net: Optional[Any] = None,
    temperature: float = 0.0,
    n_simulations: Optional[int] = None,
    depth_limit: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    selection: Optional[SelectionStrategy] = None,
    backup: Optional[BackpropStrategy] = None,
) -> List[Tuple[str, float]]:
    """
    Iteratively call ``play()`` up to ``k`` times, each call forbidding
    the first actions used by all previous calls. Returns the collected
    ``(rule_name, reward)`` tuples sorted by reward descending.

    Algorithm: branch-level diversity. After each successful ``play()``,
    the first token of the returned rule (= the root action that was
    taken first) is added to a forbidden set; the next call masks those
    children at the search root. Stops early when ``play()`` returns no
    rule (every remaining root branch is dead), so the result has at
    most ``k`` entries and at most ``len(grammar.applicable_productions
    (root))`` entries overall.

    This is a deliberately coarse diversity criterion: two different
    rules sharing the same first token are not both returned, even if
    their tails diverge. Future refinement: a Trie of forbidden full
    paths so PUCT masks at the deepest matching prefix instead of only
    the root.
    """
    if k <= 0:
        return []
    forbidden: List[str] = []
    results: List[Tuple[str, float]] = []
    for _ in range(k):
        rule, reward = play(
            log,
            grammar=grammar,
            simulator=simulator,
            net=net,
            temperature=temperature,
            n_simulations=n_simulations,
            depth_limit=depth_limit,
            rng=rng,
            forbidden_root_actions=tuple(forbidden) if forbidden else None,
            selection=selection,
            backup=backup,
        )
        if rule is None:
            break                          # ran out of live root branches
        results.append((rule, reward))
        first_token = rule.split()[0] if rule.split() else None
        if first_token is None or first_token in forbidden:
            break                          # defensive: nothing new to forbid
        forbidden.append(first_token)
    results.sort(key=lambda r: r[1], reverse=True)
    return results
