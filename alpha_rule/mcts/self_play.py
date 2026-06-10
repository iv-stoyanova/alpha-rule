"""
Self-play episode generator.

``run_self_play`` plays one episode and returns a ``Trajectory``. Starting at
the grammar's root, for each step up to ``depth_limit`` it:

    1. runs ``n_simulations`` MCTS simulations from the current node
       (select -> expand -> evaluate the leaf -> back the value up),
    2. turns the root's child visit counts into a policy ``visit_pi``,
    3. samples the next production from ``visit_pi`` and moves to that child,
    4. records a ``TrajectoryStep`` (state, policy, value target, reward).

The search loop lives here rather than in a shared helper because self-play
needs the visit distribution at the root after each round and re-roots the
tree at every chosen production.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Set

import numpy as np

from alpha_rule.evaluation.evaluator import EvalResult, Evaluator
from alpha_rule.grammar.grammar import Grammar
from alpha_rule.mcts.backprop import BackpropStrategy, MaxRewardBackup
from alpha_rule.mcts.expansion import ExpansionStrategy, RuleExpansion
from alpha_rule.mcts.node import MCTSRuleNode
from alpha_rule.mcts.replay import Trajectory, TrajectoryStep
from alpha_rule.mcts.selection import PUCTSelection, SelectionStrategy
from alpha_rule.mcts.value_target import ValueTarget, default_value_target


def _to_eval_result(raw) -> EvalResult:
    if isinstance(raw, EvalResult):
        return raw
    if isinstance(raw, tuple):
        return EvalResult(value=float(raw[0]))
    return EvalResult(value=float(raw))


def _multi_sample_chosen_reward(simulator: Evaluator, node, n: int) -> float:
    """
    Evaluate ``node`` ``n`` times and return the mean of the finite samples
    (``-inf`` if none were finite). ``n > 1`` averages out the noise of a
    stochastic simulator (a freshly-trained Q-agent per rule).
    """
    finite_sum = 0.0
    finite_count = 0
    for _ in range(max(1, n)):
        v = _to_eval_result(simulator.evaluate(node)).value
        if math.isfinite(v):
            finite_sum += v
            finite_count += 1
    return (finite_sum / finite_count) if finite_count > 0 else float("-inf")


def _normalised_visit_distribution(
    parent: MCTSRuleNode,
    *,
    temperature: float,
) -> dict:
    """
    π(a) ∝ N(a)^(1/τ); normalised. Skips dead children. Returns a dict
    mapping ``parent_action -> probability``.
    """
    live = [c for c in parent.children if not c.is_dead]
    if not live:
        return {}
    ns = np.array([float(c.N) for c in live])
    if temperature <= 0:
        # Argmax selection (deterministic).
        probs = np.zeros_like(ns)
        probs[int(ns.argmax())] = 1.0
    else:
        # Stable softmax-of-log: x = N^(1/τ) / sum.
        powered = np.power(ns, 1.0 / temperature)
        s = powered.sum()
        if s <= 0:
            probs = np.full_like(ns, 1.0 / len(ns))
        else:
            probs = powered / s
    return {c.parent_action: float(p) for c, p in zip(live, probs)}


def _sample_action(visit_pi: dict, rng: np.random.Generator) -> str:
    actions = list(visit_pi.keys())
    probs = np.array(list(visit_pi.values()), dtype=np.float64)
    probs = probs / probs.sum()                  # renormalise to be safe
    return actions[int(rng.choice(len(actions), p=probs))]


def _apply_action_to_root(
    root: MCTSRuleNode,
    action_name: str,
) -> MCTSRuleNode:
    """Find the child of ``root`` whose ``parent_action`` matches and return it."""
    for child in root.children:
        if child.parent_action == action_name:
            return child
    raise KeyError(f"action {action_name!r} not found among root's children")


def _write_priors(node: MCTSRuleNode, priors: Optional[dict]) -> None:
    """Distribute priors over node's children (by ``parent_action``)."""
    if not priors:
        return
    for child in node.children:
        if child.parent_action in priors:
            child.prior = float(priors[child.parent_action])


def _apply_root_dirichlet_noise(
    root: MCTSRuleNode,
    *,
    eps: float,
    alpha: float,
    rng: np.random.Generator,
) -> None:
    """
    Mix Dirichlet noise into the search root's child priors (AlphaZero's
    exploration trick). Only the root's own children are noised:

        prior <- (1 - eps) * prior + eps * noise,   noise ~ Dir(alpha * k)

    with ``k`` = number of live children. This keeps PUCT exploring branches it
    would otherwise starve (e.g. a child whose first visit looked bad).
    ``eps = 0`` is a no-op.

    Assumes the root is already expanded and its children's priors are set, so
    there is something to blend with. ``run_self_play`` ensures this; direct
    callers must too.
    """
    if eps <= 0.0:
        return
    live = [c for c in root.children if not getattr(c, "is_dead", False)]
    if not live:
        return
    k = len(live)
    # rng.dirichlet requires a non-zero concentration vector.
    noise = rng.dirichlet([float(alpha)] * k)
    for c, n in zip(live, noise):
        c.prior = float((1.0 - eps) * c.prior + eps * n)


def _run_one_round(
    root: MCTSRuleNode,
    *,
    n_simulations: int,
    simulator: Evaluator,
    network_evaluator: Optional[Evaluator],
    selection: SelectionStrategy,
    backup: BackpropStrategy,
    expansion: ExpansionStrategy,
    leaf_eval_mode: str = "nn",
    dead_rule_names: Optional[Set[str]] = None,
) -> None:
    """
    Run ``n_simulations`` simulations from ``root``, updating the subtree in
    place.

    ``leaf_eval_mode`` picks where a leaf's value comes from:
        ``"nn"``        network value head at non-terminal leaves, simulator at
                        terminal leaves (falls back to the simulator if there
                        is no ``network_evaluator``).
        ``"simulator"`` simulator at every leaf.

    ``dead_rule_names`` is an optional set of rule names already known to score
    ``-inf``. A newly-expanded child whose name is in the set is marked dead at
    once, so PUCT skips it and no simulator call is spent on it.
    """
    # Nothing to search if the root is fully expanded and all its children are
    # dead (e.g. every root action forbidden, or a cascade-killed subtree).
    # Without this guard each simulation would re-evaluate the root itself --
    # a wasted simulator call per simulation.
    if root.is_fully_expanded() and root.children and all(c.is_dead for c in root.children):
        return

    for _ in range(n_simulations):
        node = root

        # Selection.
        while node.is_fully_expanded() and node.children:
            picked = selection.select(node)
            if picked is None:
                break
            node = picked
        if node is None:
            continue

        # Don't expand a terminal node ("A <END>" has no continuations); it is
        # still a valid leaf to evaluate (the simulator ignores the <END>).
        if not node.is_terminal and not node.is_fully_expanded():
            new_child = expansion.expand(node)
            if new_child is None:
                continue
            node = new_child
            # If this rule already scored -inf in an earlier episode, mark it
            # dead now so no rollout spends a simulator call on it.
            if dead_rule_names and node.name in dead_rule_names:
                node.is_dead = True
            # Ask the network for priors on the parent so PUCT can use them on
            # later visits.
            if network_evaluator is not None and node.parent is not None:
                prior_result = network_evaluator.evaluate(node.parent)
                _write_priors(node.parent, prior_result.priors)

        # Evaluate the leaf and back its value up. A known-dead node scores
        # -inf with no simulator/network call.
        if getattr(node, "is_dead", False):
            backup.update(node, float("-inf"))
            continue
        is_terminal_leaf = getattr(node, "is_terminal", False)
        if (
            leaf_eval_mode == "nn"
            and not is_terminal_leaf
            and network_evaluator is not None
        ):
            result = _to_eval_result(network_evaluator.evaluate(node))
        else:
            result = _to_eval_result(simulator.evaluate(node))
        backup.update(node, result.value)


def run_self_play(
    *,
    grammar: Grammar,
    simulator: Evaluator,
    network_evaluator: Optional[Evaluator] = None,
    n_simulations: int = 50,
    depth_limit: int = 5,
    temperature: float = 1.0,
    selection: Optional[SelectionStrategy] = None,
    backup: Optional[BackpropStrategy] = None,
    rng: Optional[np.random.Generator] = None,
    n_chosen_evals: int = 1,
    dirichlet_eps: float = 0.0,
    dirichlet_alpha: float = 0.3,
    forbidden_root_actions: Optional[Iterable[str]] = None,
    leaf_eval_mode: str = "nn",
    value_target: Optional[ValueTarget] = None,
    value_scale: Optional[float] = None,
    dead_rule_names: Optional[Set[str]] = None,
) -> Trajectory:
    """
    Run one self-play episode and return its ``Trajectory``.

    Args:
        grammar: the production set / search space.
        simulator: the expensive evaluator (e.g. ``RuleSimulator``).
        network_evaluator: optional cheap evaluator supplying child priors
            (``EvalResult.priors``). Without it, priors stay at ``1.0`` and
            PUCT searches unguided.
        n_simulations: MCTS simulations per step.
        depth_limit: max construction steps (each adds one production).
        temperature: temperature for the visit-count sampling.
        selection: defaults to ``PUCTSelection()``.
        backup: defaults to ``MaxRewardBackup()``.
        rng: optional ``np.random.Generator`` for reproducibility.
        n_chosen_evals: how many simulator evaluations to average for each
            chosen-step reward (default 1). Use > 1 to average out a noisy
            simulator; ``-inf`` samples are dropped first.
        forbidden_root_actions: optional action names to forbid at the root.
            Their children are marked dead so PUCT never visits them (used to
            diversify across repeated calls).
        leaf_eval_mode: ``"nn"`` (default) uses the network value head at
            non-terminal leaves and the simulator at terminal leaves, falling
            back to the simulator if no network was given. ``"simulator"`` uses
            the simulator at every leaf.
        value_target: how each step's value target ``z_t`` is read off the
            search tree (see ``mcts.value_target``). Defaults to the strategy
            matching ``backup`` (Max -> ``MaxValue``,
            Percentile -> ``ExpectedValue``). All tree-derived targets skip
            dead children, so the value and policy targets agree on which
            branches exist.
        value_scale: positive reward cap used downstream to scale targets into
            ``[-1, +1]``. Defaults to the simulator's ``reward_scale`` if set,
            else ``None``. Stamped onto the returned ``Trajectory``.
        dead_rule_names: optional set of rule names already known to score
            ``-inf`` (e.g. gathered across episodes by ``train``). Matching
            children are marked dead on creation, so no simulator call is spent
            on them.
    """
    # Validate the Dirichlet params up front so bad input fails fast.
    if dirichlet_eps < 0.0 or dirichlet_eps > 1.0:
        raise ValueError(
            f"dirichlet_eps must be in [0, 1], got {dirichlet_eps!r}"
        )
    if dirichlet_eps > 0.0 and dirichlet_alpha <= 0.0:
        raise ValueError(
            f"dirichlet_alpha must be > 0 when dirichlet_eps > 0, "
            f"got {dirichlet_alpha!r}"
        )

    sel = selection or PUCTSelection()
    bp = backup or MaxRewardBackup()
    # Default the value target to the one that matches the backup operator so
    # the value head regresses the same quantity the search optimises.
    value_target = value_target or default_value_target(bp)
    # Auto-read the reward scale off the simulator when not given explicitly.
    if value_scale is None:
        value_scale = getattr(simulator, "reward_scale", None)
    expansion = RuleExpansion(grammar)
    rng = rng or np.random.default_rng()
    forbidden: Set[str] = (
        set(forbidden_root_actions) if forbidden_root_actions else set()
    )

    dead_set: Set[str] = (
        set(dead_rule_names) if dead_rule_names else set()
    )

    root = grammar.root()
    if forbidden:
        # Pre-expand the root and mark forbidden branches dead before any
        # rollout reaches them (PUCT skips dead children).
        while not root.is_fully_expanded():
            expansion.expand(root)
        for child in root.children:
            if child.parent_action in forbidden:
                child.is_dead = True
            elif dead_set and child.name in dead_set:
                child.is_dead = True
    steps: List[TrajectoryStep] = []
    current = root

    for depth_step in range(depth_limit):
        # Dirichlet noise on the root's child priors, once per step (skipped
        # when eps == 0, which keeps seeded runs bit-identical). Pre-expand the
        # children and set their priors first so there is something to mix.
        if dirichlet_eps > 0.0:
            while not current.is_fully_expanded():
                expansion.expand(current)
            if dead_set:
                for child in current.children:
                    if child.name in dead_set:
                        child.is_dead = True
            if network_evaluator is not None:
                prior_result = network_evaluator.evaluate(current)
                _write_priors(current, prior_result.priors)
            _apply_root_dirichlet_noise(
                current, eps=dirichlet_eps, alpha=dirichlet_alpha, rng=rng,
            )

        _run_one_round(
            current,
            n_simulations=n_simulations,
            simulator=simulator,
            network_evaluator=network_evaluator,
            selection=sel,
            backup=bp,
            expansion=expansion,
            leaf_eval_mode=leaf_eval_mode,
            dead_rule_names=dead_set if dead_set else None,
        )
        visit_pi = _normalised_visit_distribution(current, temperature=temperature)
        if not visit_pi:                              # subtree dead, stop
            break
        action_name = _sample_action(visit_pi, rng)
        next_node = _apply_action_to_root(current, action_name)

        # Reward of the chosen state from the simulator. Multi-sample
        # averaging reduces noise from the stochastic underlying evaluator
        # (see ``_multi_sample_chosen_reward``).
        chosen_reward = _multi_sample_chosen_reward(
            simulator, next_node, n_chosen_evals,
        )
        # Stamp the realised reward on the chosen node so the RealizedReturn
        # value target can read it (this node becomes next step's ``current``).
        next_node.realized_reward = chosen_reward
        applicable = tuple(
            p.name for p in grammar.applicable_productions(current)
        )
        # Per-step value target z_t at s_t, from the configured ValueTarget
        # over the search tree built above (dead children excluded; ``None``
        # -> value_targets falls back to a finite per-step default).
        state_value = value_target.state_value(current)
        steps.append(TrajectoryStep(
            state=current.name,           # s_t, policy target lives here
            visit_pi=visit_pi,
            reward=chosen_reward,         # R(s_{t+1}), describes next_node
            next_state=next_node.name,    # s_{t+1}, the rule the reward rates
            applicable_actions=applicable,  # train-time softmax mask source
            state_value=state_value,      # value target z_t at s_t
        ))
        current = next_node
        if grammar.is_terminal(current):
            break

    return Trajectory(steps=steps, value_scale=value_scale)
