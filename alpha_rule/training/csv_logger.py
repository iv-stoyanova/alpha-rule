"""
CSV logger for AlphaZero-style training runs.

One row == one training iteration; the columns reflect the AlphaZero loss
decomposition (policy + value) plus per-iteration search statistics.

File layout
-----------
    <base_dir>/<env_name>_az_<strategy>_<run>_results.csv
    <base_dir>/<env_name>_az_<strategy>_<run>_config.json

The companion JSON stores the run's hyperparameters so the
Results Comparison notebook can tag plots with the strategy and
config (c_puct, value_scale, n_simulations, ...).

Run numbers are auto-assigned by scanning existing files.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from typing import Iterable, Optional


CSV_COLUMNS = [
    "iteration",
    "wall_time_s",
    "policy_loss",
    "value_loss",
    "total_loss",
    "best_reward_in_trajectory",
    "trajectory_length",
    "n_failed_evaluations",
    "running_best_reward",
    "running_best_formula",
    "strategy",
    "activity",
    # --- v0.2.7 additions: eval + phase timings ---
    "eval_reward",
    "eval_success_rate",
    "eval_episode_length",
    "t_mcts_s",
    "t_nn_train_s",
    "t_eval_s",
    "t_buffer_s",
    # --- search-health additions ---
    "n_dead_rules",
    "buffer_fill_fraction",
    # --- value-sample harvesting ---
    "n_value_samples",
    "value_harvest_loss",
    # The rule the eval_* columns above actually describe (the play() result
    # under eval_use_play, else the running-best). Kept distinct from
    # running_best_formula so the eval numbers are never read against the
    # wrong rule.
    "eval_formula",
]


RUN_EVAL_CSV_COLUMNS = [
    "wall_time_s",
    "box_index",
    "rule_index",
    "iteration_in_rule",
    "context_rules_count",
    "running_best_formula",
    "eval_reward",
    "eval_success_rate",
    "eval_episode_length",
    "activity",
    "strategy",
    # The rule the eval_* columns describe (see CSV_COLUMNS note).
    "eval_formula",
]


def _round(value, places=6):
    """Round a float for CSV output; blank ('') for None / non-finite / NaN.

    Shared by both loggers so the two never drift on how they serialise an
    absent metric."""
    try:
        if value is None or not math.isfinite(value):
            return ""
    except TypeError:
        return ""
    return str(round(float(value), places))


def _strategy_to_filename_segment(strategy: str) -> str:
    """Normalise the strategy label so it's filesystem-safe."""
    return re.sub(r"[^\w+\-]", "_", strategy)


class AlphaZeroCSVLogger:
    """
    Append-only CSV logger for AlphaZero training iterations.

    Args:
        base_dir: directory for logs (created if missing).
        env_name: run identifier. Lives in the same directory as
            older ``_eval_N_results.csv`` logs but filenames carry
            ``_az_`` so analysis code can tell them apart.
        activity: human-readable activity tag (e.g. ``"sleep"``).
            Stored in every row for group-by plotting.
        strategy: short tag for the selection+backup combo used,
            e.g. ``"PUCT+Max"``, ``"PUCT+Percentile"``. Appears in the
            filename AND as a column so multi-strategy sweeps stay
            separable at analysis time.
        config: optional hyperparameter dict (n_simulations,
            depth_limit, c_puct, value_scale, temperature, ...).
            Saved to the companion ``_config.json``.

    Typical usage::

        logger = AlphaZeroCSVLogger(
            base_dir="...", env_name=GYM_ID, activity="sleep",
            strategy="PUCT+Max",
            config={"n_iterations": 20, "n_simulations": 50, "depth_limit": 2},
        )
        for it in training_log.iterations:
            logger.log_iteration(
                iteration=it.iteration,
                trajectory_length=it.trajectory_length,
                best_reward_in_trajectory=it.best_reward_in_trajectory,
                n_failed_evaluations=it.n_failed_evaluations,
                policy_loss=it.train_policy,
                value_loss=it.train_value,
                total_loss=it.train_total,
                best_formula=training_log.best_formula,
            )

    or the convenience one-shot::

        logger.log_training_log(training_log)
    """

    def __init__(
        self,
        base_dir: str,
        env_name: str,
        activity: str,
        *,
        strategy: str = "default",
        config: Optional[dict] = None,
    ):
        self.base_dir = base_dir
        self.env_name = env_name
        self.activity = activity
        self.strategy = strategy
        self.start_time = time.time()

        os.makedirs(base_dir, exist_ok=True)

        strategy_fs = _strategy_to_filename_segment(strategy)

        pattern = re.compile(
            rf"{re.escape(env_name)}_az_{re.escape(strategy_fs)}_(\d+)_results\.csv$"
        )
        indices = [
            int(m.group(1))
            for f in os.listdir(base_dir)
            if (m := pattern.match(f))
        ]
        self.run = max(indices, default=-1) + 1

        # Allocate the run file with exclusive-create ("x") and bump the run
        # number on collision, so two loggers started concurrently (the scan
        # above is a check-then-act race) never overwrite each other's results.
        while True:
            csv_path = os.path.join(
                base_dir, f"{env_name}_az_{strategy_fs}_{self.run}_results.csv"
            )
            try:
                f = open(csv_path, "x", newline="")
                break
            except FileExistsError:
                self.run += 1
        with f:
            csv.writer(f).writerow(CSV_COLUMNS)
        self.csv_path = csv_path
        self.config_path = os.path.join(
            base_dir, f"{env_name}_az_{strategy_fs}_{self.run}_config.json"
        )

        with open(self.config_path, "w") as f:
            json.dump(
                {
                    "env_name": env_name,
                    "activity": activity,
                    "strategy": strategy,
                    "run": self.run,
                    **(config or {}),
                },
                f,
                indent=2,
            )

        self._running_best_reward = float("-inf")
        self._running_best_formula: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Iteration-level logging
    # ------------------------------------------------------------------ #

    def log_iteration(
        self,
        *,
        iteration: int,
        trajectory_length: int,
        best_reward_in_trajectory: float,
        n_failed_evaluations: int,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
        best_formula: Optional[str] = None,
        # --- v0.2.7 additions (all optional / default-None) ---
        eval_reward: Optional[float] = None,
        eval_success_rate: Optional[float] = None,
        eval_episode_length: Optional[float] = None,
        t_mcts_s: float = 0.0,
        t_nn_train_s: float = 0.0,
        t_eval_s: float = 0.0,
        t_buffer_s: float = 0.0,
        n_dead_rules: int = 0,
        buffer_fill_fraction: Optional[float] = None,
        n_value_samples: int = 0,
        value_harvest_loss: Optional[float] = None,
        eval_formula: Optional[str] = None,
    ) -> None:
        """Append one row. Updates the running-best tracker.

        ``n_dead_rules`` is the cumulative count of rules known to fail
        (``-inf``); ``buffer_fill_fraction`` is the replay buffer's occupancy
        in ``[0, 1]``; ``eval_formula`` is the rule the ``eval_*`` metrics
        describe. All come straight from the matching ``IterationLog`` fields
        and are written verbatim."""
        # Running-best update mirrors train()'s ``log.best_formula`` rule
        # (strict reward improvement OR equal reward with a longer formula), so
        # this column never disagrees with the TrainingLog's best. Reward and
        # formula advance TOGETHER and only when a formula is present (like
        # train(), which guards the whole update on best_state is not None), so
        # the two columns can never end up describing different rules.
        if math.isfinite(best_reward_in_trajectory) and best_formula is not None:
            cand_len = len(best_formula.split())
            prev_len = (
                len(self._running_best_formula.split())
                if self._running_best_formula else -1
            )
            strict_better = best_reward_in_trajectory > self._running_best_reward
            tie_longer = (
                best_reward_in_trajectory == self._running_best_reward
                and cand_len > prev_len
            )
            if strict_better or tie_longer:
                self._running_best_reward = best_reward_in_trajectory
                self._running_best_formula = best_formula

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    iteration,
                    round(time.time() - self.start_time, 3),
                    _round(policy_loss),
                    _round(value_loss),
                    _round(total_loss),
                    _round(best_reward_in_trajectory),
                    trajectory_length,
                    n_failed_evaluations,
                    _round(self._running_best_reward),
                    self._running_best_formula or "",
                    self.strategy,
                    self.activity,
                    # --- v0.2.7 additions ---
                    _round(eval_reward),
                    _round(eval_success_rate),
                    _round(eval_episode_length),
                    _round(t_mcts_s, 4),
                    _round(t_nn_train_s, 4),
                    _round(t_eval_s, 4),
                    _round(t_buffer_s, 4),
                    n_dead_rules,
                    _round(buffer_fill_fraction, 4),
                    n_value_samples,
                    _round(value_harvest_loss, 4),
                    eval_formula or "",
                ]
            )

    def log_training_log(self, training_log, *, best_formula_override: Optional[str] = None) -> None:
        """Convenience: replay every entry from a ``TrainingLog``."""
        for it in training_log.iterations:
            self.log_iteration(
                iteration=it.iteration,
                trajectory_length=it.trajectory_length,
                best_reward_in_trajectory=it.best_reward_in_trajectory,
                n_failed_evaluations=it.n_failed_evaluations,
                policy_loss=it.train_policy,
                value_loss=it.train_value,
                total_loss=it.train_total,
                best_formula=best_formula_override or training_log.best_formula,
                eval_reward=getattr(it, "eval_reward", None),
                eval_success_rate=getattr(it, "eval_success_rate", None),
                eval_episode_length=getattr(it, "eval_episode_length", None),
                t_mcts_s=getattr(it, "t_mcts_s", 0.0),
                t_nn_train_s=getattr(it, "t_nn_train_s", 0.0),
                t_eval_s=getattr(it, "t_eval_s", 0.0),
                t_buffer_s=getattr(it, "t_buffer_s", 0.0),
                n_dead_rules=getattr(it, "n_dead_rules", 0),
                buffer_fill_fraction=getattr(it, "buffer_fill_fraction", None),
                n_value_samples=getattr(it, "n_value_samples", 0),
                value_harvest_loss=getattr(it, "value_harvest_loss", None),
                eval_formula=getattr(it, "eval_formula", None),
            )

    # ------------------------------------------------------------------ #
    # Hook helpers
    # ------------------------------------------------------------------ #

    def as_on_iteration_end(self, *, best_formula_provider=None):
        """
        Return a callback suitable for ``train(on_iteration_end=...)``.

        The callback writes one CSV row per iteration. The best formula
        comes from (in order of preference):

            1. ``best_formula_provider(it_log)`` if supplied (caller
               wants to override, e.g. tracks its own best outside the
               ``TrainingLog``).
            2. ``it_log.best_formula_in_trajectory``, the rule that
               earned this iteration's ``best_reward_in_trajectory``,
               populated by ``train()`` from ``_best_in_trajectory``.

        Without the fallback, runs using ``train()``'s default
        callback produced empty ``running_best_formula`` cells
        (pandas NaN) even though a best rule was known.
        """
        def _cb(it_log) -> None:
            bf = None
            if best_formula_provider is not None:
                bf = best_formula_provider(it_log)
            if bf is None:
                bf = getattr(it_log, "best_formula_in_trajectory", None)
            self.log_iteration(
                iteration=it_log.iteration,
                trajectory_length=it_log.trajectory_length,
                best_reward_in_trajectory=it_log.best_reward_in_trajectory,
                n_failed_evaluations=it_log.n_failed_evaluations,
                policy_loss=it_log.train_policy,
                value_loss=it_log.train_value,
                total_loss=it_log.train_total,
                best_formula=bf,
                eval_reward=getattr(it_log, "eval_reward", None),
                eval_success_rate=getattr(it_log, "eval_success_rate", None),
                eval_episode_length=getattr(it_log, "eval_episode_length", None),
                t_mcts_s=getattr(it_log, "t_mcts_s", 0.0),
                t_nn_train_s=getattr(it_log, "t_nn_train_s", 0.0),
                t_eval_s=getattr(it_log, "t_eval_s", 0.0),
                t_buffer_s=getattr(it_log, "t_buffer_s", 0.0),
                n_dead_rules=getattr(it_log, "n_dead_rules", 0),
                buffer_fill_fraction=getattr(it_log, "buffer_fill_fraction", None),
                n_value_samples=getattr(it_log, "n_value_samples", 0),
                value_harvest_loss=getattr(it_log, "value_harvest_loss", None),
                eval_formula=getattr(it_log, "eval_formula", None),
            )

        return _cb


# --------------------------------------------------------------------------- #
# Run-level eval logger (one file shared across an entire notebook run)
# --------------------------------------------------------------------------- #

class RunLevelEvalLogger:
    """
    CSV logger for RUN-WIDE eval snapshots, one row per
    ``eval_simulator.evaluate(...)`` call, aggregated across every
    rule discovered in a single notebook execution.

    Why it's separate from ``AlphaZeroCSVLogger``:
    ``AlphaZeroCSVLogger`` writes one file per rule's training (lives
    in ``per_rule_logs/``). A notebook that discovers N rules for a
    single chest produces N such files. This logger produces ONE file
    that tracks how the rule-set-so-far performs in the exploitation
    env as rules accumulate, ``box_index``, ``rule_index``, and
    ``context_rules_count`` pin which point in the discovery sequence
    each row corresponds to.

    File layout::

        <base_dir>/<env_name>_run_eval_<run>_results.csv
        <base_dir>/<env_name>_run_eval_<run>_config.json

    Typical notebook use::

        run_eval_logger = RunLevelEvalLogger(
            base_dir=base_log_dir,
            env_name=GYM_ID,
            activity="cook_breakfast",
            strategy="PUCT+Max",
            config={"eval_every": 5, ...},
        )

        for box_idx in range(n_boxes):
            for rule_idx in range(N_RULES_PER_BOX):
                run_eval_logger.set_context(
                    box_index=box_idx,
                    rule_index=rule_idx,
                    context_rules_count=len(best_rules),
                )
                train(
                    ...,
                    on_iteration_end=compose(
                        per_rule_logger.as_on_iteration_end(),
                        run_eval_logger.as_on_iteration_end(),
                    ),
                )
                best_rules.append(log.best_formula)
    """

    def __init__(
        self,
        base_dir: str,
        env_name: str,
        activity: str,
        *,
        strategy: str = "default",
        config: Optional[dict] = None,
    ):
        self.base_dir = base_dir
        self.env_name = env_name
        self.activity = activity
        self.strategy = strategy
        self.start_time = time.time()

        os.makedirs(base_dir, exist_ok=True)

        pattern = re.compile(
            rf"{re.escape(env_name)}_run_eval_(\d+)_results\.csv$"
        )
        indices = [
            int(m.group(1))
            for f in os.listdir(base_dir)
            if (m := pattern.match(f))
        ]
        self.run = max(indices, default=-1) + 1

        # Exclusive-create the run file (bump on collision) so concurrent
        # loggers never clobber each other -- see AlphaZeroCSVLogger.__init__.
        while True:
            csv_path = os.path.join(
                base_dir, f"{env_name}_run_eval_{self.run}_results.csv"
            )
            try:
                f = open(csv_path, "x", newline="")
                break
            except FileExistsError:
                self.run += 1
        with f:
            csv.writer(f).writerow(RUN_EVAL_CSV_COLUMNS)
        self.csv_path = csv_path
        self.config_path = os.path.join(
            base_dir, f"{env_name}_run_eval_{self.run}_config.json"
        )

        with open(self.config_path, "w") as f:
            json.dump(
                {
                    "env_name": env_name,
                    "activity": activity,
                    "strategy": strategy,
                    "run": self.run,
                    **(config or {}),
                },
                f,
                indent=2,
            )

        self._box_index: Optional[int] = None
        self._rule_index: Optional[int] = None
        self._context_rules_count: int = 0

    # ------------------------------------------------------------------ #
    # Layout helpers
    # ------------------------------------------------------------------ #

    @property
    def per_rule_dir(self) -> str:
        """Canonical sub-folder for rule-level logs of this run-eval.

        Used by the activity notebooks to wire up the companion
        ``AlphaZeroCSVLogger`` so its CSVs land next to their
        corresponding run-eval CSV. The directory is NOT created by
        this property, the caller does that when needed.
        """
        return os.path.join(self.base_dir, f"run_eval_{self.run}")

    # ------------------------------------------------------------------ #
    # Context setters (called once per rule discovery)
    # ------------------------------------------------------------------ #

    def set_context(
        self,
        *,
        box_index: int,
        rule_index: int,
        context_rules_count: int = 0,
    ) -> None:
        """Call before each ``train()`` so the subsequent rows are
        tagged with (box, rule) indices and the current context size."""
        self._box_index = box_index
        self._rule_index = rule_index
        self._context_rules_count = context_rules_count

    # ------------------------------------------------------------------ #
    # Row-append API
    # ------------------------------------------------------------------ #

    def log_eval(
        self,
        *,
        iteration_in_rule: int,
        running_best_formula: Optional[str],
        eval_reward: Optional[float],
        eval_success_rate: Optional[float],
        eval_episode_length: Optional[float],
        eval_formula: Optional[str] = None,
    ) -> None:
        """Append one eval row. Caller must have called ``set_context`` first.

        ``running_best_formula`` is the training-side best (for context);
        ``eval_formula`` is the rule the ``eval_*`` numbers actually describe
        (they differ under ``eval_use_play``)."""
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    round(time.time() - self.start_time, 3),
                    self._box_index if self._box_index is not None else "",
                    self._rule_index if self._rule_index is not None else "",
                    iteration_in_rule,
                    self._context_rules_count,
                    running_best_formula or "",
                    _round(eval_reward),
                    _round(eval_success_rate),
                    _round(eval_episode_length),
                    self.activity,
                    self.strategy,
                    eval_formula or "",
                ]
            )

    # ------------------------------------------------------------------ #
    # train() callback
    # ------------------------------------------------------------------ #

    def as_on_iteration_end(self):
        """
        Return an ``on_iteration_end`` callback that writes ONE row per
        iteration ONLY when eval data is present (``it_log.eval_reward
        is not None``). Iterations where the eval didn't fire (gated by
        ``eval_every``) produce no row.
        """
        def _cb(it_log) -> None:
            if getattr(it_log, "eval_reward", None) is None:
                return
            self.log_eval(
                iteration_in_rule=it_log.iteration,
                running_best_formula=getattr(it_log, "best_formula_in_trajectory", None),
                eval_reward=it_log.eval_reward,
                eval_success_rate=getattr(it_log, "eval_success_rate", None),
                eval_episode_length=getattr(it_log, "eval_episode_length", None),
                eval_formula=getattr(it_log, "eval_formula", None),
            )

        return _cb


# --------------------------------------------------------------------------- #
# Loaders for the Results Comparison notebook
# --------------------------------------------------------------------------- #

def load_alphazero_logs(base_dir: str):
    """
    Load every ``*_az_<strategy>_<run>_results.csv`` in ``base_dir``.

    Returns a pandas DataFrame (requires pandas at call time, not a
    hard dependency of this module) with columns:

        <CSV_COLUMNS>  + file, run, strategy_fs, label,
        config_<key>   for each key in the companion ``_config.json``

    ``label`` is ``"<activity> [<strategy>] (run <run>)"`` and is
    suitable for direct use as a legend entry.
    """
    import pandas as pd                                # lazy import

    pattern = re.compile(
        r"(?P<env>.+?)_az_(?P<strategy_fs>.+?)_(?P<run>\d+)_results\.csv$"
    )

    # Walk the whole tree, supports the flat older layout
    # (``base_dir/*.csv`` or ``base_dir/per_rule_logs/*.csv``) and the
    # new per-strategy nested layout
    # (``base_dir/<strategy>/run_eval_<N>/*.csv``) transparently.
    frames = []
    discovered = []
    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            m = pattern.match(fname)
            if m:
                discovered.append((root, fname, m))
    discovered.sort(key=lambda t: (t[0], t[1]))

    for root, fname, m in discovered:
        env_name = m.group("env")
        strategy_fs = m.group("strategy_fs")
        run = int(m.group("run"))
        csv_path = os.path.join(root, fname)
        config_path = os.path.join(root, f"{env_name}_az_{strategy_fs}_{run}_config.json")

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Flatten config into config_* columns. Non-scalar config values
        # (lists, dicts, tuples) get serialised to JSON strings so
        # pandas can broadcast them across the DataFrame's rows, 
        # otherwise assigning an empty list ``[]`` to a 20-row frame
        # raises "Length of values (0) does not match length of index (20)".
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                if k in ("env_name", "activity", "strategy", "run"):
                    continue
                if isinstance(v, (list, dict, tuple)):
                    v = json.dumps(v)
                df[f"config_{k}"] = v

        df["file"] = fname
        df["run"] = run
        df["strategy_fs"] = strategy_fs
        df["label"] = (
            df["activity"].astype(str)
            + " [" + df["strategy"].astype(str) + "]"
            + " (run " + df["run"].astype(str) + ")"
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS + ["file", "run", "strategy_fs", "label"])
    return pd.concat(frames, ignore_index=True)


def load_run_level_eval_logs(base_dir: str):
    """
    Load every ``*_run_eval_<run>_results.csv`` in ``base_dir``.

    Returns a pandas DataFrame with columns:

        <RUN_EVAL_CSV_COLUMNS>  + file, run, label,
        config_<key>   for each key in the companion ``_config.json``

    ``label`` is ``"<activity> [<strategy>] (run <run>)"`` and is
    suitable for direct use as a legend entry.
    """
    import pandas as pd                                # lazy import

    pattern = re.compile(
        r"(?P<env>.+?)_run_eval_(?P<run>\d+)_results\.csv$"
    )

    # Recursive walk for the same reasons as ``load_alphazero_logs``.
    frames = []
    discovered = []
    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            m = pattern.match(fname)
            if m:
                discovered.append((root, fname, m))
    discovered.sort(key=lambda t: (t[0], t[1]))

    for root, fname, m in discovered:
        env_name = m.group("env")
        run = int(m.group("run"))
        csv_path = os.path.join(root, fname)
        config_path = os.path.join(
            root, f"{env_name}_run_eval_{run}_config.json"
        )

        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        # Flatten config the same way as load_alphazero_logs (serialise
        # list / dict / tuple values to JSON strings so pandas can
        # broadcast them).
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                if k in ("env_name", "activity", "strategy", "run"):
                    continue
                if isinstance(v, (list, dict, tuple)):
                    v = json.dumps(v)
                df[f"config_{k}"] = v

        df["file"] = fname
        df["run"] = run
        df["label"] = (
            df["activity"].astype(str)
            + " [" + df["strategy"].astype(str) + "]"
            + " (run " + df["run"].astype(str) + ")"
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=RUN_EVAL_CSV_COLUMNS + ["file", "run", "label"]
        )
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "AlphaZeroCSVLogger",
    "CSV_COLUMNS",
    "RUN_EVAL_CSV_COLUMNS",
    "RunLevelEvalLogger",
    "load_alphazero_logs",
    "load_run_level_eval_logs",
]
