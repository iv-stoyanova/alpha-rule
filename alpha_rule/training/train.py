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
from alpha_rule.mcts.replay import ReplayBuffer, Trajectory, ValueBuffer
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

    n_value_samples: int = 0
    """Value-head samples harvested from the search tree this iteration
    (``len(Trajectory.value_samples)``; 0 when no ``value_sample_collector``
    ran)."""

    value_harvest_loss: float = 0.0
    """Mean value-only MSE across this iteration's ``train_value_step`` calls on
    the harvested samples (0.0 when no value-only training ran)."""

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
    eval_formula: Optional[str] = None
    """The rule the ``eval_*`` metrics above actually describe: the
    ``play()`` result when ``eval_use_play=True``, otherwise
    ``log.best_formula``. ``None`` on iterations where the eval didn't fire.
    Logged next to the eval metrics so they are never read against the wrong
    rule (under ``eval_use_play`` it differs from
    ``best_formula_in_trajectory``)."""

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
    neg_value_scale: Optional[float] = None
    """Negative reward cap used for asymmetric value scaling during training.
    ``None`` means it mirrored ``value_scale`` (symmetric). ``play()`` reuses it
    so its ``NeuralEvaluator`` de-scales the value head exactly as training
    did."""
    selection: Optional[SelectionStrategy] = None
    """The exact ``SelectionStrategy`` training used (the resolved PUCT object).
    ``play()`` defaults to it so a bare ``play(log, ...)`` reproduces the
    training-time search instead of falling back to library defaults."""
    backup: Optional[BackpropStrategy] = None
    """The exact ``BackpropStrategy`` training used. ``play()`` defaults to it
    for the same reason as ``selection``."""
    normalizer: Optional[Any] = None
    """The read-time ``RewardNormalizer`` training used (``None`` when
    ``normalize=False``). ``play()`` reuses it so the rollout normalizes Q and
    de-scales the value head as training did."""
    norm_k: float = 2.0
    """Std-per-unit for the normalizer (see ``RewardNormalizer``)."""
    end_prior_scale: float = 1.0
    """Multiplier on the terminal (<END>) prior used at training time. ``play()``
    reuses it so its ``NeuralEvaluator`` down-weights <END> the same way."""


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
    if isinstance(raw, tuple):
        # Take whatever the tuple provides, leaving the rest ``None``; an empty
        # tuple yields no reward (NaN -> blank in the logs) rather than crashing.
        reward = float(raw[0]) if len(raw) >= 1 else float("nan")
        success = float(raw[1]) if len(raw) >= 2 else None
        steps = float(raw[2]) if len(raw) >= 3 else None
        return reward, success, steps
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
    neg_value_scale: Optional[float] = None,
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
    fpu_baseline: float = float("-inf"),
    selection: Optional[SelectionStrategy] = None,
    # --- backpropagation ------------------------------------------- #
    backup: str | BackpropStrategy = "max",
    percentile: float = 20.0,
    min_samples: int = 10,
    value_target=None,
    value_sample_collector=None,
    value_train_steps: int = 0,
    # --- exploration / leaf evaluation ----------------------------- #
    dirichlet_eps: float = 0.25,
    dirichlet_alpha: float = 0.3,
    leaf_eval_mode: str = "nn",
    leaf_eval_warmup: int = 0,
    n_chosen_evals: int = 1,
    # --- read-time reward normalization ---------------------------- #
    normalize: bool = True,
    norm_k: float = 2.0,
    norm_robust: bool = True,
    end_prior_scale: float = 1.0,
    # --- evaluation / logging / misc ------------------------------- #
    eval_simulator: Optional[Evaluator] = None,
    eval_every: int = 5,
    eval_use_play: bool = False,
    on_iteration_end=None,
    device: Optional[str] = None,
    seed: int = 0,
    debug: int = 0,
    debug_every: int = 1,
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
        debug: verbosity of the live training trace (default 0, off). ``1``
            prints a one-line summary per iteration (trajectory length, the
            best step and running-best formula with rewards, the three losses,
            dead-rule count, buffer fill, and the eval metrics on eval
            iterations) plus the self-play path of every episode. ``2`` also
            prints the full PUCT option table at every construction step. ``3``
            also traces every MCTS simulation (round header, per-selection PUCT
            scores and pick, expansions, per-leaf eval mode + value, dead /
            forbidden marking); very verbose (see ``run_self_play``'s ``debug``).
            Purely observational.
        debug_every: when ``debug >= 2``, restrict the expensive per-step PUCT
            tables to iterations where ``iteration % debug_every == 0`` (default
            1, every iteration). The per-iteration summary and the self-play
            path still print every iteration; only the verbose tables are
            thinned. Ignored when ``debug < 2``.
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
        neg_value_scale: negative reward cap for ASYMMETRIC value scaling.
            ``None`` (default) mirrors ``value_scale`` (symmetric, historical
            behaviour). Set it larger than ``value_scale`` when the positive cap
            is small (e.g. ``value_scale=3`` for 3 boxes) but the penalty tail
            runs much more negative: good rules then keep full resolution in
            ``[0, 1]`` while bad rules spread across ``[-1, 0)`` instead of all
            saturating at ``-1``. Threaded identically into the replay buffer,
            the ``NeuralEvaluator`` de-scaling, and the stored ``TrainingLog``.
        leaf_eval_mode: ``"nn"`` (default, AlphaZero leaf bootstrap)
            uses the network's value head at non-terminal leaves and
            the simulator only at terminal leaves; ``"simulator"`` uses
            the simulator at every leaf. The chosen-step reward written
            to the trajectory is always from the simulator regardless of
            this flag.
        leaf_eval_warmup: number of initial iterations that score every
            leaf with the simulator before switching to ``leaf_eval_mode``.
            Default ``0`` (off). Withholds the untrained value head, which
            early on predicts over-negative on promising nodes and steers
            the search away from them. Set ``>= buffer_warmup`` so the net
            has trained on the warmup trajectories before the switch.
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
        fpu_baseline: lower bound on the first-play-urgency value an UNVISITED
            child can receive when building the default ``PUCTSelection``
            (ignored when an explicit ``selection`` is passed). Default
            ``-inf`` is a no-op; a finite value (e.g. ``0.0``) stops a parent
            dragged very negative by bad sibling backups from suppressing
            exploration of fresh actions.
        percentile, min_samples: ``PercentileRewardBackup`` knobs, used
            only when ``backup="percentile"``.
        value_target: which quantity the value head regresses, as a name
            (``"max"`` = ``Q_max``, best rule reachable from a state;
            ``"expected"``, ``"mean_percentile"``, ``"realized"``), a
            ``ValueTarget`` instance, or ``None``/``"auto"`` (default: the one
            matching ``backup`` -- Max->MaxValue, Percentile->ExpectedValue).
            Use ``"max"`` so a state's target reflects the best completion
            found beneath it rather than the mean of all rollouts through it.
        value_sample_collector: optional ``ValueSampleCollector`` (see
            ``mcts.value_collect``). When set, after each self-play search the
            tree is harvested for extra ``(state, value)`` samples (e.g.
            ``TreeQmaxCollector`` emits every explored node's ``Q_max``), which
            train the value head only. ``None`` (default) = no harvest.
        value_train_steps: number of value-head-only gradient steps per
            iteration on the harvested samples (``train_value_step``). Gated only
            on the value buffer being non-empty (independent of ``buffer_warmup``).
            Default ``0`` = off (no value-only training even if a collector runs).

    Returns:
        ``TrainingLog`` with per-iteration metrics and the best
        formula encountered.
    """
    # Lazy heavy imports so test discovery doesn't pay the torch cost.
    import torch
    from alpha_rule.evaluation.neural_evaluator import NeuralEvaluator
    from alpha_rule.nn.model import AllenFormulaNet
    from alpha_rule.nn.tokenizer import GrammarTokenizer
    from alpha_rule.nn.training import train_step, train_value_step

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
    # Read-time reward normalizer, one per train() call. Shared by selection,
    # the value target, and the NeuralEvaluator de-scale; None disables it.
    from alpha_rule.mcts.normalize import RewardNormalizer
    normalizer = RewardNormalizer(robust=norm_robust) if normalize else None
    # Wire the evaluator's value scale to the simulator's reward cap (an explicit
    # value_scale wins) so the network value and the simulator reward share one
    # scale in the MCTS backup.
    if value_scale is not None:
        network_evaluator = NeuralEvaluator(
            model, grammar, max_len=max_len, value_scale=value_scale,
            neg_value_scale=neg_value_scale,
            normalizer=normalizer, norm_k=norm_k,
            end_prior_scale=end_prior_scale,
        )
    else:
        network_evaluator = NeuralEvaluator.from_simulator(
            model, grammar, expensive_simulator, max_len=max_len,
            neg_value_scale=neg_value_scale,
            normalizer=normalizer, norm_k=norm_k,
            end_prior_scale=end_prior_scale,
        )
    resolved_scale = network_evaluator.value_scale
    resolved_neg_scale = network_evaluator.neg_value_scale
    buffer = ReplayBuffer(
        capacity=buffer_capacity,
        value_scale=resolved_scale,
        neg_value_scale=resolved_neg_scale,
    )
    # Value-head-only buffer for samples harvested from the search tree. Stays
    # empty unless a value_sample_collector is given.
    value_buffer = ValueBuffer(capacity=buffer_capacity)

    # Resolve the search strategies once. ``selection``/``backup`` accept an
    # explicit object (advanced override, takes precedence); otherwise they
    # are built from the scalar kwargs so every knob is visible at this call.
    sel = selection if selection is not None else PUCTSelection(
        c_puct=c_puct, fpu_reduction=fpu_reduction, q_source=q_source,
        fpu_baseline=fpu_baseline, normalizer=normalizer, norm_k=norm_k,
    )
    # Attach the normalizer to an explicit selection that lacks one, so its Q
    # and the de-scaled NN leaf value share a scale.
    if (
        selection is not None
        and normalizer is not None
        and getattr(sel, "normalizer", None) is None
    ):
        sel.normalizer = normalizer
        sel.norm_k = norm_k
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

    # Resolve the value-target selector: a name ("max"/"expected"/
    # "mean_percentile"/"realized"), a ValueTarget instance, or None/"auto"
    # (= the target matching the backup, picked by default_value_target in
    # run_self_play). "max" trains the value head on Q_max (best rule reachable
    # from a state) instead of the backup's default mean.
    if value_target is None or isinstance(value_target, str):
        from alpha_rule.mcts.value_target import (
            ExpectedValue, MaxValue, MeanPercentileValue, RealizedReturn,
        )
        _vt_map = {
            "max": MaxValue, "expected": ExpectedValue,
            "mean_percentile": MeanPercentileValue, "mean": MeanPercentileValue,
            "realized": RealizedReturn,
        }
        if value_target is None or value_target == "auto":
            resolved_value_target = None
        elif value_target in _vt_map:
            resolved_value_target = _vt_map[value_target]()
        else:
            raise ValueError(
                f"unknown value_target: {value_target!r}; expected one of "
                f"{sorted(_vt_map)}, 'auto', or a ValueTarget instance"
            )
    else:
        resolved_value_target = value_target

    # --- Risky / inconsistent configuration warnings ----------------- #
    # Non-fatal: surface combinations that silently misbehave so a long run
    # doesn't crash mid-search or train against a mis-scaled signal.
    _warn_risky_config(
        max_len=max_len,
        depth_limit=depth_limit,
        # value_scale is unused under normalization, so skip its mismatch check.
        explicit_value_scale=(None if normalize else value_scale),
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
        neg_value_scale=resolved_neg_scale,
        selection=sel,
        backup=bp,
        normalizer=normalizer,
        norm_k=norm_k,
        end_prior_scale=end_prior_scale,
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

        # Score every leaf with the simulator during the warmup window, then
        # switch to the configured leaf_eval_mode once the net has trained.
        leaf_eval_mode_it = (
            "simulator" if it < leaf_eval_warmup else leaf_eval_mode
        )

        # --- MCTS / self-play -------------------------------------- #
        # Resolve this iteration's self-play debug level: paths + chosen-node
        # lines (level 1) print every iteration when debug is on; the verbose
        # per-step PUCT tables (level 2) are thinned to every ``debug_every``.
        if not debug:
            sp_debug = 0
        elif it % max(1, debug_every) == 0:
            sp_debug = debug
        else:
            sp_debug = min(debug, 1)
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
            leaf_eval_mode=leaf_eval_mode_it,
            value_target=resolved_value_target,
            value_sample_collector=value_sample_collector,
            value_scale=resolved_scale,
            neg_value_scale=resolved_neg_scale,
            normalizer=normalizer,
            norm_k=norm_k,
            dead_rule_names=dead_rules if dead_rules else None,
            debug=sp_debug,
            debug_tag=f"it={it}",
        )
        t_mcts_s = time.perf_counter() - t0

        # Count this episode's failed (-inf) evaluations from the raw
        # trajectory BEFORE folding those rules into the persistent dead set,
        # so the metric reflects what this episode actually saw independent of
        # the dead-rule bookkeeping below.
        n_failed = _failed_count(traj)

        # Accumulate -inf rules seen in this episode into the persistent
        # set so future iterations can short-circuit them: both the committed
        # steps that scored -inf (``_collect_dead_rules``) and the rule nodes
        # killed mid-search because their ``<END>`` never fired
        # (``traj.dead_names``, which are no longer committed so the former would
        # miss them).
        dead_rules.update(_collect_dead_rules(traj))
        dead_rules.update(traj.dead_names)

        # --- Replay buffer push ------------------------------------ #
        t0 = time.perf_counter()
        buffer.push_trajectory(traj)
        if traj.value_samples:
            value_buffer.push(traj.value_sample_targets())
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
        # Value-head-only training on harvested tree samples. Gated on the value
        # buffer (not buffer_warmup), so it can start on the first episode.
        n_value_samples = len(traj.value_samples)
        value_harvest_loss = 0.0
        if value_train_steps > 0 and len(value_buffer) > 0:
            _v_sum = 0.0
            for _ in range(value_train_steps):
                _v_sum += train_value_step(
                    model, optimizer, value_buffer.sample(batch_size),
                    max_len=max_len, grad_clip=grad_clip,
                )
            value_harvest_loss = _v_sum / max(1, value_train_steps)
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
        eval_formula: Optional[str] = None
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
                        n_chosen_evals=n_chosen_evals,
                        leaf_eval_mode=leaf_eval_mode_it,
                        dead_rule_names=dead_rules if dead_rules else None,
                    )
                    rule_to_eval = play_rule
                except Exception:
                    # Defensive: if play() fails (e.g. empty buffer edge
                    # cases on the first iterations) fall back to the
                    # default path rather than bringing down training.
                    rule_to_eval = log.best_formula
            else:
                rule_to_eval = log.best_formula
            # Record WHICH rule the eval_* metrics describe so they are never
            # read against best_formula_in_trajectory (a different rule under
            # eval_use_play).
            eval_formula = rule_to_eval
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
            n_value_samples=n_value_samples,
            value_harvest_loss=value_harvest_loss,
            best_formula_in_trajectory=best_state_str,
            eval_reward=eval_reward,
            eval_success_rate=eval_success,
            eval_episode_length=eval_steps,
            eval_formula=eval_formula,
            t_mcts_s=t_mcts_s,
            t_nn_train_s=t_nn_train_s,
            t_eval_s=t_eval_s,
            t_buffer_s=t_buffer_s,
        )
        log.iterations.append(it_log)

        # One-line live trace per iteration (off unless debug). Sits next to
        # the self-play path that run_self_play already printed for this
        # episode, so the run reads top-to-bottom as "what got searched" then
        # "where the run stands".
        if debug:
            br = it_log.best_reward_in_trajectory
            brtxt = f"{br:.2f}" if math.isfinite(br) else "-inf"
            rbtxt = f"{log.best_reward:.2f}" if math.isfinite(log.best_reward) else "-inf"
            line = (
                f"[it={it}] len={it_log.trajectory_length} "
                f"best_step={it_log.best_formula_in_trajectory!r} ({brtxt}) | "
                f"running_best={log.best_formula!r} ({rbtxt}) | "
                f"loss tot={it_log.train_total:.3f} pol={it_log.train_policy:.3f} "
                f"val={it_log.train_value:.3f} | dead={it_log.n_dead_rules} "
                f"buf={it_log.buffer_fill_fraction:.2f}"
            )
            if it_log.n_value_samples or it_log.value_harvest_loss:
                line += (
                    f" | harvest={it_log.n_value_samples} "
                    f"v_loss={it_log.value_harvest_loss:.3f}"
                )
            if it_log.eval_reward is not None:
                stxt = (
                    f"{it_log.eval_success_rate:.2f}"
                    if it_log.eval_success_rate is not None else "n/a"
                )
                line += (
                    f" | EVAL {it_log.eval_formula!r} "
                    f"r={it_log.eval_reward:.2f} succ={stxt}"
                )
            print(line)
            # At debug>=2, list exactly which nodes were harvested this
            # iteration (name=raw Q_max), best-reachable first.
            if debug >= 2 and traj.value_samples:
                rows = sorted(traj.value_samples, key=lambda kv: kv[1], reverse=True)
                shown = ", ".join(f"{n}={q:.2f}" for n, q in rows[:30])
                extra = f"  (+{len(rows) - 30} more)" if len(rows) > 30 else ""
                print(f"  [it={it}] harvested {len(rows)} nodes: {shown}{extra}")

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
    dead_rule_names: Optional[Iterable[str]] = None,
    n_chosen_evals: int = 1,
    selection: Optional[SelectionStrategy] = None,
    backup: Optional[BackpropStrategy] = None,
    leaf_eval_mode: str = "nn",
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
        dead_rule_names: optional set of full rule names to treat as dead
            (the same mechanism MCTS uses for ``-inf`` rules): any node whose
            name is in the set is marked dead on expansion, so the search will
            not commit to it but every prefix and sibling stays live. Used by
            ``play_top_k`` for rule-level diversity.
        n_chosen_evals: simulator samples averaged for each chosen-step
            reward (default 1). Pass the training value to match the noise
            handling used during training.
        selection: optional ``SelectionStrategy``. Defaults to the one
            ``train()`` used (``log.selection``), so a bare ``play(log, ...)``
            reproduces the training-time search instead of falling back to
            library defaults. Pass an explicit object to override.
        backup: optional ``BackpropStrategy``. Defaults to ``log.backup``;
            same rationale as ``selection``.

    Returns:
        ``(rule_name, reward)``. ``rule_name`` is ``None`` and
        ``reward`` is ``-inf`` only when the rollout produced no steps
        at all (e.g., every root child was already dead).
    """
    import math

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

    # Default the search strategies to the exact ones training used (stored on
    # the log) so a bare play(log, ...) reproduces training-time MCTS instead of
    # silently falling back to library defaults; an explicit arg still wins.
    sel = selection if selection is not None else log.selection
    bp = backup if backup is not None else log.backup

    # Reuse the scale the training run resolved (stored on the log) so the
    # network value comes back in the simulator's reward units; fall back to the
    # simulator's own reward_scale, then to raw passthrough.
    scale = log.value_scale if log.value_scale is not None else getattr(
        simulator, "reward_scale", None
    )
    neg_scale = log.neg_value_scale     # None -> NeuralEvaluator mirrors `scale`
    # Reuse the training normalizer (read-only) so play de-scales the value head
    # as training did; None when training was unnormalized.
    if scale is not None:
        network_evaluator = NeuralEvaluator(
            model, grammar, max_len=log.max_len, value_scale=scale,
            neg_value_scale=neg_scale,
            normalizer=log.normalizer, norm_k=log.norm_k,
            end_prior_scale=log.end_prior_scale,
        )
    else:
        network_evaluator = NeuralEvaluator.from_simulator(
            model, grammar, simulator, max_len=log.max_len,
            neg_value_scale=neg_scale,
            normalizer=log.normalizer, norm_k=log.norm_k,
            end_prior_scale=log.end_prior_scale,
        )

    # No torch.no_grad() needed: every network call goes through
    # model.predict(), which already runs under torch.inference_mode().
    traj = run_self_play(
        grammar=grammar,
        simulator=simulator,
        network_evaluator=network_evaluator,
        n_simulations=sims,
        depth_limit=depth,
        temperature=temperature,
        selection=sel,
        backup=bp,
        rng=rng,
        n_chosen_evals=n_chosen_evals,
        leaf_eval_mode=leaf_eval_mode,
        forbidden_root_actions=forbidden_root_actions,
        dead_rule_names=set(dead_rule_names) if dead_rule_names else None,
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
    n_chosen_evals: int = 1,
    selection: Optional[SelectionStrategy] = None,
    backup: Optional[BackpropStrategy] = None,
) -> List[Tuple[str, float]]:
    """
    Call ``play()`` up to ``k`` times, each call forbidding the exact rules
    already found, and return the collected ``(rule_name, reward)`` tuples
    sorted by reward descending.

    Diversity is rule-level: after each successful ``play()`` the full rule
    string is added to a forbidden set passed back in as ``dead_rule_names``
    (the same dead-node mechanism MCTS uses for ``-inf`` rules). The next
    search marks that exact rule dead so it cannot be re-committed, while every
    prefix and sibling stays live. This avoids the failure mode of forbidding
    whole root branches: with a 2-letter alphabet scored by the count of ``A``
    and depth 3, the top rule ``"A A A"`` would forbid the entire ``A`` branch
    and the next rule ``"B A A"`` the ``B`` branch, leaving no live root action
    by the third call. Forbidding the exact rule instead lets the second call
    return ``"A A B"`` and so on.

    Stops early when ``play()`` returns no rule (the reachable rule space is
    exhausted), so the result has at most ``k`` entries.
    """
    if k <= 0:
        return []
    found: set = set()
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
            dead_rule_names=found if found else None,
            n_chosen_evals=n_chosen_evals,
            selection=selection,
            backup=backup,
        )
        if rule is None:
            break                          # reachable rule space exhausted
        if rule in found:
            break                          # defensive: play() repeated a forbidden rule
        found.add(rule)
        results.append((rule, reward))
    results.sort(key=lambda r: r[1], reverse=True)
    return results
