"""
Tests for ``alpha_rule.training.RunLevelEvalLogger`` and its companion
loader ``load_run_level_eval_logs``.

Pins:
    - writes the expected header + companion ``_config.json``
    - auto-numbered run index (scans existing files)
    - ``set_context`` values persist across subsequent ``log_eval`` calls
    - ``as_on_iteration_end`` callback SKIPS rows whose ``eval_reward``
      is ``None`` (eval didn't fire that iteration)
    - ``load_run_level_eval_logs`` reads everything back with flattened
      ``config_*`` columns and a tidy ``label``
"""
from __future__ import annotations

import csv
import json
import os
import tempfile

from alpha_rule.training.csv_logger import (
    RUN_EVAL_CSV_COLUMNS,
    RunLevelEvalLogger,
    load_run_level_eval_logs,
)
from alpha_rule.training.train import IterationLog


# --------------------------------------------------------------------------- #
# File layout + auto-numbering
# --------------------------------------------------------------------------- #

def test_run_level_eval_logger_writes_header_and_config():
    with tempfile.TemporaryDirectory() as td:
        logger = RunLevelEvalLogger(
            base_dir=td,
            env_name="ENV-v0_sleep",
            activity="sleep",
            strategy="PUCT+Max",
            config={"eval_every": 5, "context_rules": []},
        )
        with open(logger.csv_path) as f:
            header = next(csv.reader(f))
        assert header == RUN_EVAL_CSV_COLUMNS
        with open(logger.config_path) as f:
            cfg = json.load(f)
        assert cfg["strategy"] == "PUCT+Max"
        assert cfg["activity"] == "sleep"
        assert cfg["eval_every"] == 5


def test_run_level_eval_logger_auto_numbered_runs():
    with tempfile.TemporaryDirectory() as td:
        a = RunLevelEvalLogger(td, "ENV", "sleep")
        b = RunLevelEvalLogger(td, "ENV", "sleep")
        c = RunLevelEvalLogger(td, "ENV", "sleep")
        assert a.run == 0
        assert b.run == 1
        assert c.run == 2


# --------------------------------------------------------------------------- #
# Row-append API
# --------------------------------------------------------------------------- #

def test_set_context_and_log_eval_rows():
    with tempfile.TemporaryDirectory() as td:
        logger = RunLevelEvalLogger(td, "ENV", "sleep", strategy="PUCT+Max")

        logger.set_context(box_index=0, rule_index=0, context_rules_count=0)
        logger.log_eval(
            iteration_in_rule=5, running_best_formula="A",
            eval_reward=0.3, eval_success_rate=0.1, eval_episode_length=40.0,
        )
        logger.log_eval(
            iteration_in_rule=10, running_best_formula="A B <",
            eval_reward=0.6, eval_success_rate=0.4, eval_episode_length=12.0,
        )

        logger.set_context(box_index=0, rule_index=1, context_rules_count=1)
        logger.log_eval(
            iteration_in_rule=5, running_best_formula="C D >",
            eval_reward=0.8, eval_success_rate=0.6, eval_episode_length=8.0,
        )

        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert int(rows[0]["rule_index"]) == 0
        assert int(rows[2]["rule_index"]) == 1
        assert int(rows[2]["context_rules_count"]) == 1
        assert rows[0]["running_best_formula"] == "A"
        assert rows[2]["running_best_formula"] == "C D >"
        assert float(rows[1]["eval_reward"]) == 0.6


# --------------------------------------------------------------------------- #
# as_on_iteration_end: skip rows without eval
# --------------------------------------------------------------------------- #

def test_as_on_iteration_end_skips_iterations_without_eval():
    with tempfile.TemporaryDirectory() as td:
        logger = RunLevelEvalLogger(td, "ENV", "sleep", strategy="PUCT+Max")
        logger.set_context(box_index=0, rule_index=0, context_rules_count=0)
        cb = logger.as_on_iteration_end()

        # Iteration with eval (should be logged)
        cb(IterationLog(
            iteration=0, trajectory_length=2,
            best_reward_in_trajectory=0.9, n_failed_evaluations=0,
            train_total=0.1, train_policy=0.05, train_value=0.05,
            best_formula_in_trajectory="A B <",
            eval_reward=0.4, eval_success_rate=0.2, eval_episode_length=20.0,
        ))
        # Iteration without eval (should be skipped)
        cb(IterationLog(
            iteration=1, trajectory_length=2,
            best_reward_in_trajectory=0.5, n_failed_evaluations=0,
            train_total=0.1, train_policy=0.05, train_value=0.05,
            best_formula_in_trajectory="A B <",
            # eval_* defaulted to None → callback should NOT append a row
        ))

        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert int(rows[0]["iteration_in_rule"]) == 0
        assert float(rows[0]["eval_reward"]) == 0.4


# --------------------------------------------------------------------------- #
# load_run_level_eval_logs
# --------------------------------------------------------------------------- #

def test_load_run_level_eval_logs_empty_dir_returns_empty_dataframe():
    with tempfile.TemporaryDirectory() as td:
        df = load_run_level_eval_logs(td)
        assert len(df) == 0


def test_load_run_level_eval_logs_merges_config_and_rows():
    with tempfile.TemporaryDirectory() as td:
        logger = RunLevelEvalLogger(
            td, "ENV-v0_sleep", "sleep", strategy="PUCT+Max",
            config={"eval_every": 5, "context_rules": ["A", "B"]},
        )
        logger.set_context(box_index=0, rule_index=0, context_rules_count=2)
        logger.log_eval(
            iteration_in_rule=5, running_best_formula="A B <",
            eval_reward=0.6, eval_success_rate=0.3, eval_episode_length=20.0,
        )
        logger.log_eval(
            iteration_in_rule=10, running_best_formula="A B < C <",
            eval_reward=0.9, eval_success_rate=0.5, eval_episode_length=10.0,
        )

        df = load_run_level_eval_logs(td)
        assert len(df) == 2
        assert set(df["activity"].unique()) == {"sleep"}
        assert set(df["strategy"].unique()) == {"PUCT+Max"}
        assert "config_eval_every" in df.columns
        assert "config_context_rules" in df.columns
        # List value was JSON-stringified for broadcast safety
        assert df["config_context_rules"].iloc[0] == '["A", "B"]'
        assert "label" in df.columns
        assert "sleep" in df["label"].iloc[0]


def test_load_run_level_eval_logs_walks_nested_directories():
    """
    Mirrors the per-rule loader's nested-layout test. With the new
    per-strategy layout ``logs/<strategy>/<env>_run_eval_<N>.csv``,
    calling the loader on the top-level ``logs/`` must recursively
    discover CSVs under every strategy sub-folder.
    """
    with tempfile.TemporaryDirectory() as td:
        strat_a = os.path.join(td, "PUCT+Max")
        strat_b = os.path.join(td, "PUCT+Max-GT")
        os.makedirs(strat_a); os.makedirs(strat_b)

        la = RunLevelEvalLogger(
            strat_a, "ENV-v0_sleep", "sleep", strategy="PUCT+Max",
            config={"eval_every": 5},
        )
        la.set_context(box_index=0, rule_index=0, context_rules_count=0)
        la.log_eval(
            iteration_in_rule=5, running_best_formula="A <",
            eval_reward=0.6, eval_success_rate=0.3, eval_episode_length=20.0,
        )

        lb = RunLevelEvalLogger(
            strat_b, "ENV-v0_sleep", "sleep", strategy="PUCT+Max-GT",
            config={"eval_every": 5},
        )
        lb.set_context(box_index=0, rule_index=0, context_rules_count=0)
        lb.log_eval(
            iteration_in_rule=5, running_best_formula="B <",
            eval_reward=0.4, eval_success_rate=0.1, eval_episode_length=15.0,
        )

        df = load_run_level_eval_logs(td)
        assert len(df) == 2
        assert set(df["strategy"].unique()) == {"PUCT+Max", "PUCT+Max-GT"}


def test_run_level_eval_logger_per_rule_dir_property():
    """
    ``per_rule_dir`` returns the canonical sub-folder name and does NOT
    create it (caller is responsible for mkdir).
    """
    with tempfile.TemporaryDirectory() as td:
        logger = RunLevelEvalLogger(
            td, "ENV-v0_sleep", "sleep", strategy="PUCT+Max",
        )
        expected = os.path.join(td, f"run_eval_{logger.run}")
        assert logger.per_rule_dir == expected
        assert not os.path.exists(expected)
