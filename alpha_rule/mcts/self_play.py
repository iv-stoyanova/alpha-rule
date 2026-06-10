"""
Self-play episode generator.

One call produces one ``Trajectory`` of ``(state, visit_pi, reward)``
tuples by:

    1. Starting at the root (an empty / start node).
    2. For each construction step until ``depth_limit``:
        a. Run ``n_simulations`` MCTS simulations from the current node.
           - Selection uses the supplied ``selection`` (default PUCT).
           - Backprop uses the supplied ``backup`` (default Max).
           - Leaf rewards come from ``simulator.evaluate``.
           - If ``network_evaluator`` is provided AND the leaf was a
             newly-expanded child, its priors are written onto sibling
             nodes so PUCT can use them on subsequent visits.
        b. Compute ``visit_pi`` from the root's children's visit counts
           with temperature ``temperature``.
        c. Sample the next production proportional to ``visit_pi``.
        d. Apply the production, evaluate the new state's reward,
           record the trajectory step.
    3. Return the ``Trajectory``.

The search loop is kept local (rather than a generic helper) because
self-play needs both (a) the visit distribution at the root after each
round of simulations and (b) progressive replanting of the root after
each chosen production.
"""
from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Set

import numpy as np

from alpha_rule.evaluation.evaluator import EvalResult, Evaluator
from alpha_rule.grammar.grammar import Grammar
from alpha_rule.mcts.backprop import BackpropStrategy, MaxRewardBackup
from alpha_rule.mcts.expansion import ExpansionStrategy, RuleExpansion
from alpha_rule.mcts.node import MCTSRuleNode
from alpha_rule.mcts.replay import DEFAULT_REWARD_FLOOR, Trajectory, TrajectoryStep
from alpha_rule.mcts.selection import PUCTSelection, SelectionStrategy


def _debug_enabled() -> bool:
    """Diagnostic prints are gated by the ``MCTS_DEBUG`` env var."""
    return False


def _to_eval_result(raw) -> EvalResult:
    if isinstance(raw, EvalResult):
        return raw
    if isinstance(raw, tuple):
        return EvalResult(value=float(raw[0]))
    return EvalResult(value=float(raw))


def _multi_sample_chosen_reward(simulator: Evaluator, node, n: int) -> float:
    """
    Evaluate ``node`` ``n`` times and return the mean of the finite
    samples (``-inf`` if every sample was non-finite). ``n > 1`` reduces
    variance from a stochastic simulator (e.g. a freshly-trained
    Q-learning agent per rule).
    """
    finite_sum = 0.0
    finite_count = 0
    for _ in range(max(1, n)):
        v = _to_eval_result(simulator.evaluate(node)).value
        if math.isfinite(v):
            finite_sum += v
            finite_count += 1
    return (finite_sum / finite_count) if finite_count > 0 else float("-inf")


def _compute_root_value(
    parent: MCTSRuleNode,
    *,
    dead_penalty: float = DEFAULT_REWARD_FLOOR,
) -> Optional[float]:
    """
    MCTS root value at ``parent``: visit-weighted average of children's
    ``Q_max`` over the rollouts that completed from ``parent`` so far.

    Dead-aware: children whose ``Q_max`` is ``-inf`` (every observed
    rollout from that subtree was a failure) contribute ``dead_penalty``
    instead of ``-inf`` so the average stays finite and the value head
    can learn "this state has dead branches mean lower value". Without the
    substitution the average would be ``-inf``, breaking MSE.

    Returns ``None`` when no usable statistic exists yet, either no
    child has been visited or the only visited children produced a
    non-finite weighted sum even after substitution. Callers should
    treat ``None`` as "fall back to a different target" rather than
    propagate it into the NN.
    """
    visited = [c for c in parent.children if c.N > 0]
    total_n = sum(c.N for c in visited)
    if total_n <= 0:
        return None
    weighted_sum = 0.0
    for c in visited:
        q = c.Q_max if math.isfinite(c.Q_max) else float(dead_penalty)
        weighted_sum += c.N * q
    raw = weighted_sum / total_n
    return raw if math.isfinite(raw) else None


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
    probs = probs / probs.sum()                  # paranoid renorm
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
    AlphaZero-style Dirichlet noise on the MCTS search-ROOT's children
    priors. Leaves subtree priors untouched, only the root node itself
    gets noised, which is the canonical recipe from Silver et al.

        p_new[a] = (1 - eps) * p_old[a] + eps * noise[a]
        noise ~ Dir([alpha] * k)   where k = #live children.

    Enables residual exploration of branches PUCT would otherwise
    permanently starve (e.g. a child whose first visit returned a very
    negative Q). ``eps = 0`` is a no-op.

    This function assumes ``root.children`` has already been populated
    (i.e. the root has been fully expanded) and its children's priors
    have been written, so the mix has something meaningful to blend
    with. ``run_self_play`` calls this between rounds; callers who use
    it directly should ensure the same invariant.
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
    Run ``n_simulations`` MCTS simulations rooted at ``root``. Mutates
    ``root`` and its subtree in place.

    ``leaf_eval_mode`` controls where the leaf value comes from:
        ``"nn"``: network's value head at non-terminal leaves;
            ``simulator`` only at terminal leaves. Falls back to the
            simulator silently when ``network_evaluator`` is ``None``.
        ``"simulator"``: simulator at every leaf.

    ``dead_rule_names`` (optional): set of rule-name strings already
    known to evaluate to ``-inf`` from a prior episode. Any new child
    whose ``name`` matches is marked ``is_dead=True`` immediately, so
    PUCT never visits it. Saves a full ``simulator.evaluate`` call (a
    Q-learning training run) per revisit.
    """
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

        # Expansion. Skip for terminal nodes: expanding "A <END>" into
        # "A <END> <" is semantically incoherent (the rule already
        # ended). The terminal itself is still a valid simulation
        # target, simulator.evaluate strips the <END> marker.
        if not node.is_terminal and not node.is_fully_expanded():
            new_child = expansion.expand(node)
            if new_child is None:
                continue
            node = new_child
            # Cross-episode dead-rule masking. If this rule was seen to
            # return -inf in a previous self-play episode, mark it dead
            # before any rollout reaches it. PUCTSelection.score short-
            # circuits is_dead children, and the dead-ancestor cascade
            # in MaxRewardBackup handles propagation upward when every
            # sibling is dead.
            if dead_rule_names and node.name in dead_rule_names:
                node.is_dead = True
            # Ask the network for priors on the parent (which now has
            # this newly-expanded child plus possibly siblings) so PUCT
            # can use them on later visits.
            if network_evaluator is not None and node.parent is not None:
                prior_result = network_evaluator.evaluate(node.parent)
                _write_priors(node.parent, prior_result.priors)

        # Simulation. AlphaZero-style: at non-terminal leaves the value
        # comes from the network's value head; at terminal leaves we
        # still pay the cost of a real ``simulator.evaluate`` so MCTS
        # gets ground-truth grounding at the most informative state.
        # Short-circuit dead rules, both the simulator and the network
        # would burn cycles on a rule whose -inf outcome is already known.
        if getattr(node, "is_dead", False):
            backup.update(node, float("-inf"))
            if _debug_enabled():
                print(f"[eval] {node.name!r} -> SKIPPED (known dead)", flush=True)
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
        if _debug_enabled():
            print(f"[eval] {node.name!r} -> value={result.value:+.6f}", flush=True)


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
    reward_floor: float = DEFAULT_REWARD_FLOOR,
    dead_rule_names: Optional[Set[str]] = None,
) -> Trajectory:
    """
    Run one self-play episode and return its ``Trajectory``.

    Args:
        grammar: production set.
        simulator: expensive evaluator (e.g. ``RuleSimulator``).
        network_evaluator: optional cheap evaluator that supplies priors
            (``EvalResult.priors``) on newly-expanded nodes. If None,
            child priors stay at the default ``1.0`` and PUCT degenerates
            toward an unguided (prior-free) search.
        n_simulations: MCTS simulations per construction step.
        depth_limit: max number of construction steps. (Each step adds
            one production, equivalent to a level on the tree.)
        temperature: softmax temperature for visit-count sampling.
        selection: defaults to ``PUCTSelection()``.
        backup: defaults to ``MaxRewardBackup()``.
        rng: optional ``np.random.Generator`` for reproducibility.
        n_chosen_evals: number of independent ``simulator.evaluate``
            samples to average when computing the chosen-step reward
            (default 1). Larger values reduce variance from the
            underlying stochastic simulator (e.g., a freshly-trained
            Q-learning agent per rule); non-finite samples (``-inf``)
            are dropped before averaging.
        forbidden_root_actions: optional set of action names that may
            not be taken at the search root. Implementation: the root is
            pre-expanded and every child whose ``parent_action`` is in
            this set is marked ``is_dead = True`` so PUCT never visits
            it. Used by ``play_top_k`` to enforce branch-level diversity
            across iterative ``play()`` calls. Future refinement: full
            path-prefix masking (Trie of forbidden prefixes) for finer
            diversity inside a branch.
        leaf_eval_mode: ``"nn"`` (default) uses the network's value head
            at non-terminal leaves and ``simulator`` at terminal leaves, 
            the AlphaZero leaf-bootstrap. Falls back to ``simulator`` if
            no ``network_evaluator`` was supplied. ``"simulator"`` uses
            the expensive simulator at every leaf, terminal or not.
        reward_floor: value substituted for a dead child's ``-inf``
            ``Q_max`` when computing the per-step MCTS root value, so the
            value target stays finite. Should match the
            ``ReplayBuffer.reward_floor`` used downstream (default
            ``-100.0``).
        dead_rule_names: optional set of rule-name strings already known
            to evaluate to ``-inf`` (e.g., accumulated across previous
            ``run_self_play`` calls inside ``train()``). Whenever MCTS
            expansion produces a child whose ``name`` is in the set, the
            child is marked ``is_dead=True`` so PUCT skips it and no
            simulator call is ever issued for it. Pre-expanded root
            children (Dirichlet / ``forbidden_root_actions`` paths) are
            also masked.
    """
    # Validate Dirichlet params early so the reject-bad-input tests
    # don't have to wait for the first round to trip a deep assertion.
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
        # Pre-expand the root and kill forbidden branches before any MCTS
        # visits them. PUCT skips ``is_dead=True`` children, so this fully
        # masks them.
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
        # AlphaZero-style Dirichlet noise on the search root's children.
        # Pre-expand children and populate priors so the mix has something
        # to blend with; noise is sampled once per construction step and
        # persists for this depth_step's ``n_simulations`` rollouts. Skip
        # when ``dirichlet_eps == 0`` to preserve bit-exact reproducibility
        # of runs on older branches.
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
        if _debug_enabled():
            label = "root" if depth_step == 0 else f"depth={depth_step}"
            print(f"[self_play] {label} node={current.name!r} children:", flush=True)
            for child in current.children:
                dead = " DEAD" if getattr(child, "is_dead", False) else ""
                q = child.Q_max if np.isfinite(child.Q_max) else float("nan")
                print(
                    f"  action={child.parent_action!r:<40s} "
                    f"prior={child.prior:.4f}  N={child.N:4d}  "
                    f"Q_max={q:+.4f}{dead}",
                    flush=True,
                )
        visit_pi = _normalised_visit_distribution(current, temperature=temperature)
        if not visit_pi:                              # subtree dead, stop
            if _debug_enabled():
                print(f"[self_play] depth={depth_step} subtree dead, stopping", flush=True)
            break
        action_name = _sample_action(visit_pi, rng)
        next_node = _apply_action_to_root(current, action_name)

        # Reward of the chosen state. Use the same simulator. (Two-tier
        # variants would use the cheap eval instead in some conditions.)
        # Multi-sample averaging reduces noise from the stochastic
        # underlying evaluator (see ``_multi_sample_chosen_reward``).
        chosen_reward = _multi_sample_chosen_reward(
            simulator, next_node, n_chosen_evals,
        )
        if _debug_enabled():
            print(
                f"[self_play] depth={depth_step} chose action={action_name!r} "
                f"-> next={next_node.name!r}  chosen_reward={chosen_reward:+.6f}",
                flush=True,
            )
        applicable = tuple(
            p.name for p in grammar.applicable_productions(current)
        )
        # Capture the MCTS root value at s_t from the search-tree
        # statistics built by the rollouts above. Dead-aware: dead
        # children's -inf Q_max gets replaced by the reward floor so
        # the average stays finite. None propagates a fallback to
        # value_targets (NN never sees -inf/NaN).
        root_value = _compute_root_value(current, dead_penalty=reward_floor)
        steps.append(TrajectoryStep(
            state=current.name,           # s_t, policy target lives here
            visit_pi=visit_pi,
            reward=chosen_reward,         # R(s_{t+1}), describes next_node
            next_state=next_node.name,    # s_{t+1}, the rule the reward rates
            applicable_actions=applicable,  # train-time softmax mask source
            root_value=root_value,        # ExIt-style value target at s_t
        ))
        current = next_node
        if grammar.is_terminal(current):
            break

    # print(f"depth limit: {depth_limit}")
    # print(f"Trajectory steps: {steps}")
    return Trajectory(steps=steps)
