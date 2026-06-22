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


def _multi_sample_chosen_reward(
    simulator: Evaluator, node, n: int,
) -> float:
    """
    Evaluate ``node`` ``n`` times and return the mean of the finite samples
    (``-inf`` if none were finite).

    With a self-seeding simulator (``resample_seed``) the ``n`` calls are
    independent draws; with a deterministic one they are identical, so ``n=1``
    is the usual setting.
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


def _debug_print_options(
    parent: MCTSRuleNode,
    selection: SelectionStrategy,
    chosen_action: str,
    *,
    tag: str,
    depth: int,
) -> None:
    """Print the PUCT table for ``parent``'s children, sorted by PUCT score.

    Columns: N (visit count), Q_max (best value backed up), Q_fmean (the
    percentile-filtered mean ``Q_sum / N_passers``), prior (network P(s, a)),
    PUCT (``selection.score``). The chosen child is flagged ``*``, a dead child
    ``x``.
    """
    sum_n = sum(c.N for c in parent.children)
    print(f"  [{tag} d={depth}] PUCT options at {parent.name!r}  (sum_N={sum_n}):")
    rows = []
    for c in parent.children:
        try:
            puct = float(selection.score(parent, c))
        except Exception:
            puct = float("nan")
        q_fmean = (c.Q_sum / c.N_passers) if c.N_passers > 0 else float("nan")
        rows.append((c, puct, q_fmean))
    rows.sort(key=lambda r: (r[1] if math.isfinite(r[1]) else -1e18), reverse=True)
    for c, puct, q_fmean in rows:
        mark = "*" if c.parent_action == chosen_action else ("x" if c.is_dead else " ")
        qmax = "  -inf" if c.Q_max == float("-inf") else f"{c.Q_max:6.2f}"
        qfm = "   nan" if not math.isfinite(q_fmean) else f"{q_fmean:6.2f}"
        puct = "  -inf" if not math.isfinite(puct) else f"{puct:7.2f}"
        print(f"    {mark} {str(c.parent_action):<16} N={c.N:<4d} "
              f"Q_max={qmax} Q_fmean={qfm} prior={c.prior:.3f} PUCT={puct}")


def _debug_print_path(path: List[tuple], *, tag: str) -> None:
    """Print the production-by-production path this episode committed to:
    ``<ROOT>`` then one ``-[action]-> rule (r=reward)`` line per chosen step,
    closing with the best finite-reward step seen along it."""
    print(f"  [{tag}] self-play path:")
    print(f"      <ROOT>")
    best_name, best_r = None, float("-inf")
    for action, rule, reward in path:
        rtxt = f"{reward:6.2f}" if math.isfinite(reward) else "  -inf"
        print(f"        -[{action}]-> {rule}   (r={rtxt})")
        if math.isfinite(reward) and reward > best_r:
            best_name, best_r = rule, reward
    btxt = f"{best_r:.2f}" if math.isfinite(best_r) else "-inf"
    print(f"      best finite step: {best_name!r}  (r={btxt})")


def _debug_print_diag(
    parent: MCTSRuleNode,
    chosen_action: str,
    chosen_node: MCTSRuleNode,
    *,
    tag: str,
    depth: int,
) -> None:
    """Print two diagnostics for the committed step: whether the chosen node is
    dead (it should never be, since selection and the visit-count commit both
    skip dead children) alongside the dead-child count, and whether the
    ``END_RULE`` child was expanded and how it ranked.
    """
    children = parent.children
    n_dead = sum(1 for c in children if getattr(c, "is_dead", False))
    end = next((c for c in children if getattr(c, "is_terminal", False)), None)
    if end is None:
        end_txt = "END unexpanded (N=0, never selected -> cannot be committed)"
    else:
        flag = "CHOSEN" if end.parent_action == chosen_action else "not chosen"
        end_txt = (f"END expanded N={end.N} prior={end.prior:.3f} "
                   f"dead={end.is_dead} ({flag})")
    print(f"  [{tag} d={depth}] diag: chosen is_dead="
          f"{getattr(chosen_node, 'is_dead', False)} | "
          f"children {len(children)}/{parent.n_possible_actions} expanded "
          f"(dead {n_dead}) | {end_txt}")


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
    depth_limit: Optional[int] = None,
    newly_dead: Optional[Set[str]] = None,
    normalizer=None,
    q_trace_collector=None,
    debug: int = 0,
    debug_tag: str = "",
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

    ``depth_limit`` caps the search depth: a node at ``level == depth_limit``
    is a valid leaf to evaluate but is never expanded deeper. The committed
    trajectory only ever reaches ``depth_limit`` productions, so searching past
    it evaluates rules that can never be the final answer, wasting simulator /
    network calls and (with a tight ``max_len``) overflowing the tokenizer mid
    search. ``None`` leaves the search uncapped.
    """
    # Nothing to search if the root is fully expanded and all its children are
    # dead (e.g. every root action forbidden, or a cascade-killed subtree).
    # Without this guard each simulation would re-evaluate the root itself --
    # a wasted simulator call per simulation.
    if root.is_fully_expanded() and root.children and all(c.is_dead for c in root.children):
        return

    for sim_i in range(n_simulations):
        node = root

        # Selection.
        while node.is_fully_expanded() and node.children:
            if debug >= 3:
                scored = sorted(
                    ((c.parent_action, selection.score(node, c)) for c in node.children),
                    key=lambda kv: (kv[1] if math.isfinite(kv[1]) else -1e18),
                    reverse=True,
                )
                shown = ", ".join(
                    f"{a}={'-inf' if not math.isfinite(s) else f'{s:.2f}'}"
                    for a, s in scored[:8]
                )
                print(f"  [{debug_tag}] sim{sim_i} select at {node.name!r}: {shown}"
                      + (" ..." if len(scored) > 8 else ""))
            picked = selection.select(node)
            if picked is None:
                break
            if debug >= 3:
                print(f"  [{debug_tag}] sim{sim_i}   -> picked {picked.name!r} "
                      f"(N={picked.N})")
            node = picked
        if node is None:
            continue

        # Don't expand a terminal node ("A <END>" has no continuations); it is
        # still a valid leaf to evaluate (the simulator ignores the <END>).
        # Also stop at depth_limit: a node already at the construction budget is
        # evaluated as a leaf but never grown deeper (see the depth_limit note
        # in the docstring).
        can_expand = depth_limit is None or node.level < depth_limit
        if not node.is_terminal and not node.is_fully_expanded() and can_expand:
            new_child = expansion.expand(node)
            if new_child is None:
                continue
            node = new_child
            if debug >= 3:
                print(f"  [{debug_tag}] sim{sim_i} expand -> {node.name!r}")
            # If this rule already scored -inf in an earlier episode, mark it
            # dead now so no rollout spends a simulator call on it.
            if dead_rule_names and node.name in dead_rule_names:
                node.is_dead = True
                if debug >= 3:
                    print(f"  [{debug_tag}] sim{sim_i}   marked dead "
                          f"(in dead set): {node.name!r}")
            # Ask the network for priors on the parent so PUCT can use them on
            # later visits.
            if network_evaluator is not None and node.parent is not None:
                prior_result = network_evaluator.evaluate(node.parent)
                _write_priors(node.parent, prior_result.priors)

        # Evaluate the leaf and back its value up. A known-dead node scores
        # -inf with no simulator/network call.
        if getattr(node, "is_dead", False):
            backup.update(node, float("-inf"))
            if debug >= 3:
                print(f"  [{debug_tag}] sim{sim_i} leaf {node.name!r} DEAD -> -inf")
            continue
        is_terminal_leaf = getattr(node, "is_terminal", False)
        if (
            leaf_eval_mode == "nn"
            and not is_terminal_leaf
            and network_evaluator is not None
        ):
            result = _to_eval_result(network_evaluator.evaluate(node))
            mode = "nn"
        else:
            result = _to_eval_result(simulator.evaluate(node))
            mode = "sim"
            # Update the normalizer with simulator rewards only; NN leaf values
            # are estimates, not fresh rewards.
            if normalizer is not None and math.isfinite(result.value):
                normalizer.update(result.value)
        if debug >= 3:
            v = result.value
            vtxt = f"{v:.2f}" if math.isfinite(v) else ("-inf" if v < 0 else "nan")
            print(f"  [{debug_tag}] sim{sim_i} eval {node.name!r} "
                  f"({'terminal' if is_terminal_leaf else 'leaf'}) "
                  f"via {mode} -> {vtxt}")
        backup.update(node, result.value)
        if q_trace_collector is not None:
            q_trace_collector.record_eval(node, result.value, mode)

        # A terminal <END> scoring -inf means the rule never fires. The grammar
        # is monotone (extensions only add constraints), so no extension can
        # fire either and the whole subtree is dead. Mark the rule node (the
        # <END>'s parent) dead so the search stops committing it this episode,
        # and record its name so train() prunes it in later iterations. Only the
        # rule node dies; its ancestors and siblings are untouched.
        if (
            is_terminal_leaf
            and result.value == float("-inf")
            and node.parent is not None
            and not node.parent.is_dead
        ):
            node.parent.is_dead = True
            if newly_dead is not None:
                newly_dead.add(node.parent.name)
            if debug >= 3:
                print(f"  [{debug_tag}] sim{sim_i}   rule {node.parent.name!r} "
                      f"never fires -> marked dead (subtree pruned)")


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
    value_sample_collector=None,
    value_scale: Optional[float] = None,
    neg_value_scale: Optional[float] = None,
    normalizer=None,
    norm_k: float = 2.0,
    q_trace_collector=None,
    decision_trace_collector=None,
    iteration: int = 0,
    dead_rule_names: Optional[Set[str]] = None,
    debug: int = 0,
    debug_tag: str = "",
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
        temperature: temperature for the action SAMPLING only. The policy
            target stored in each step (``visit_pi``) is always the tau=1 visit
            fractions, independent of this value, so an annealed sampling
            temperature does not distort the training signal. ``temperature=0``
            samples the argmax visit (greedy play); ``1.0`` samples
            proportional to visits.
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
        neg_value_scale: negative reward cap for asymmetric value scaling.
            ``None`` (default) mirrors ``value_scale`` (symmetric). Stamped onto
            the returned ``Trajectory`` so the buffer scales targets to match.
        dead_rule_names: optional set of rule names already known to score
            ``-inf`` (e.g. gathered across episodes by ``train``). Matching
            children are marked dead on creation, so no simulator call is spent
            on them.
        debug: verbosity of the per-step diagnostic prints (default 0, off).
            ``1`` prints, for each committed construction step, which child was
            chosen with its value, and at the end the full self-play path. ``2``
            additionally prints the PUCT option table at every step (every
            child's N / Q_max / filtered-mean Q / prior / PUCT score), i.e. the
            full search landscape the choice was made over. ``3`` additionally
            traces every MCTS simulation inside the round: the round header
            (depth + node), each selection step with its candidate PUCT scores
            and the pick, each expansion, each leaf evaluation with its mode
            (``nn``/``sim``) and value, and any dead/forbidden marking. Very
            verbose; prints only, never changes what the search does.
        debug_tag: short label prefixed to every debug line (e.g. ``"it=3"``),
            so interleaved output from many episodes stays attributable.
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
    # Negative-side cap for asymmetric value scaling; defaults to the positive
    # cap (symmetric, historical behaviour) when not given.
    if neg_value_scale is None:
        neg_value_scale = value_scale
    expansion = RuleExpansion(grammar)
    rng = rng or np.random.default_rng()
    forbidden: Set[str] = (
        set(forbidden_root_actions) if forbidden_root_actions else set()
    )

    dead_set: Set[str] = (
        set(dead_rule_names) if dead_rule_names else set()
    )

    if debug >= 3 and (forbidden or dead_set):
        print(f"  [{debug_tag}] removing sets -> forbidden={sorted(forbidden)} "
              f"dead={sorted(dead_set)}")

    root = grammar.root()
    if forbidden:
        # Pre-expand the root and mark forbidden branches dead before any
        # rollout reaches them (PUCT skips dead children).
        while not root.is_fully_expanded():
            expansion.expand(root)
        for child in root.children:
            if child.parent_action in forbidden:
                child.is_dead = True
                if debug >= 3:
                    print(f"  [{debug_tag}] forbidden -> dead: {child.name!r}")
            elif dead_set and child.name in dead_set:
                child.is_dead = True
                if debug >= 3:
                    print(f"  [{debug_tag}] dead-set -> dead: {child.name!r}")
    steps: List[TrajectoryStep] = []
    current = root
    debug_path: List[tuple] = []
    # Rule names killed mid-episode (a <END> scored -inf). Stamped on the
    # returned Trajectory so train() folds them into its persistent dead set.
    newly_dead: Set[str] = set()
    # Value samples harvested from each per-step search tree (name -> max raw
    # target), deduped across steps keeping the best estimate. Empty unless a
    # value_sample_collector was given.
    harvested: dict = {}

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

        if debug >= 3:
            print(f"  [{debug_tag}] === round at depth={depth_step} "
                  f"node={current.name!r} (level={current.level}, N={current.N}) "
                  f"x{n_simulations} sims ===")
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
            depth_limit=depth_limit,
            newly_dead=newly_dead,
            normalizer=normalizer,
            q_trace_collector=q_trace_collector,
            debug=debug,
            debug_tag=debug_tag,
        )
        # Harvest extra value-head samples from the tree just built (e.g. each
        # explored node's Q_max), keeping the max per name across steps.
        if value_sample_collector is not None:
            for name, raw in value_sample_collector.collect(current):
                prev = harvested.get(name)
                if prev is None or raw > prev:
                    harvested[name] = raw
        # Fold rules killed this round into the live dead set so later steps of
        # this same episode skip them on expansion too.
        if newly_dead:
            dead_set.update(newly_dead)
        # Policy TARGET: the raw visit fractions at tau=1, the AlphaZero
        # training signal stored in the trajectory and regressed by the policy
        # head. It is computed independently of the action-sampling temperature
        # below, so annealing the choice toward argmax never collapses the
        # stored target to a one-hot (which would train the net to be
        # over-confident about whichever branch the search happened to commit to).
        target_pi = _normalised_visit_distribution(current, temperature=1.0)
        if not target_pi:                             # subtree dead, stop
            break
        # Action SAMPLING distribution at the requested (possibly annealed)
        # temperature: tau == 1 reuses the target as-is; tau -> 0 sharpens
        # toward the argmax visit.
        if temperature == 1.0:
            sample_pi = target_pi
        else:
            sample_pi = _normalised_visit_distribution(current, temperature=temperature)
        action_name = _sample_action(sample_pi, rng)
        next_node = _apply_action_to_root(current, action_name)
        if decision_trace_collector is not None:
            decision_trace_collector.record_decision(
                iteration, depth_step, current, action_name)

        # Mean simulator reward for the chosen state (n independent draws when the
        # simulator self-seeds via resample_seed; n=1 by default).
        chosen_reward = _multi_sample_chosen_reward(
            simulator, next_node, n_chosen_evals,
        )
        # Stamp the realised reward (kept raw) on the chosen node for the
        # RealizedReturn value target; this node becomes next step's ``current``.
        next_node.realized_reward = chosen_reward
        if normalizer is not None and math.isfinite(chosen_reward):
            normalizer.update(chosen_reward)

        # Diagnostic prints (off by default). Level 2 dumps the full PUCT
        # landscape this step chose over; level 1 logs just the chosen child
        # and its value. Observational only; no effect on the search.
        if debug >= 2:
            _debug_print_options(
                current, sel, action_name, tag=debug_tag, depth=depth_step,
            )
        if debug >= 1:
            rtxt = f"{chosen_reward:.2f}" if math.isfinite(chosen_reward) else "-inf"
            qfm = (next_node.Q_sum / next_node.N_passers) if next_node.N_passers > 0 else float("nan")
            qfmtxt = f"{qfm:.2f}" if math.isfinite(qfm) else "nan"
            qmaxtxt = "-inf" if next_node.Q_max == float("-inf") else f"{next_node.Q_max:.2f}"
            print(f"  [{debug_tag} d={depth_step}] chose {action_name!r} "
                  f"-> {next_node.name!r}  N={next_node.N} "
                  f"Q_max={qmaxtxt} Q_fmean={qfmtxt} reward={rtxt}")
            _debug_print_diag(
                current, action_name, next_node, tag=debug_tag, depth=depth_step,
            )
        debug_path.append((action_name, next_node.name, chosen_reward))

        applicable = tuple(
            p.name for p in grammar.applicable_productions(current)
        )
        # Per-step value target z_t at s_t, from the configured ValueTarget
        # over the search tree built above (dead children excluded; ``None``
        # -> value_targets falls back to a finite per-step default).
        state_value = value_target.state_value(current)
        steps.append(TrajectoryStep(
            state=current.name,           # s_t, policy target lives here
            visit_pi=target_pi,           # tau=1 visit fractions (decoupled from sampling)
            reward=chosen_reward,         # R(s_{t+1}), describes next_node
            next_state=next_node.name,    # s_{t+1}, the rule the reward rates
            applicable_actions=applicable,  # train-time softmax mask source
            state_value=state_value,      # value target z_t at s_t
        ))
        current = next_node
        if grammar.is_terminal(current):
            break

    if debug >= 1:
        _debug_print_path(debug_path, tag=debug_tag)

    return Trajectory(
        steps=steps, value_scale=value_scale, neg_value_scale=neg_value_scale,
        norm_mean=(normalizer.mean if normalizer is not None else None),
        norm_std=(normalizer.std if normalizer is not None else None),
        norm_k=(norm_k if normalizer is not None else None),
        dead_names=sorted(newly_dead),
        value_samples=list(harvested.items()),
    )
