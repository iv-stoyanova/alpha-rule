"""
Tests for ``training.csv_logger.AlphaZeroCSVLogger``.

Pins:
    - Auto-numbered run (scans existing matching files in the dir).
    - Header + per-iteration row written with the expected columns.
    - ``-inf`` best_reward_in_trajectory renders as an empty cell,
      NOT as "-inf" (which would break pandas.to_numeric readers
      downstream).
    - Running-best tracker keeps the highest finite reward seen.
    - ``log_training_log`` iterates a ``TrainingLog`` correctly.
    - Companion ``_config.json`` is written with the user-supplied
      hyperparameters.
    - ``load_alphazero_logs`` reads back what was written.
"""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile

from alpha_rule.training.csv_logger import (
    AlphaZeroCSVLogger,
    CSV_COLUMNS,
    load_alphazero_logs,
)
from alpha_rule.training.train import IterationLog, TrainingLog


# --------------------------------------------------------------------------- #
# Per-iteration logging
# --------------------------------------------------------------------------- #

def test_logger_writes_header_and_config():
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(
            base_dir=td,
            env_name="ENV-v0_sleep",
            activity="sleep",
            strategy="PUCT+Max",
            config={"n_simulations": 50, "depth_limit": 2},
        )
        # CSV exists with expected header
        with open(logger.csv_path) as f:
            header = next(csv.reader(f))
        assert header == CSV_COLUMNS
        # Config companion exists and contains user kwargs
        with open(logger.config_path) as f:
            cfg = json.load(f)
        assert cfg["strategy"] == "PUCT+Max"
        assert cfg["activity"] == "sleep"
        assert cfg["n_simulations"] == 50


def test_log_iteration_emits_one_row_with_finite_values():
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep", strategy="PUCT+Max")
        logger.log_iteration(
            iteration=3,
            trajectory_length=4,
            best_reward_in_trajectory=0.75,
            n_failed_evaluations=2,
            policy_loss=1.5,
            value_loss=0.3,
            total_loss=1.8,
            best_formula="A B <",
        )
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        r = rows[0]
        assert int(r["iteration"]) == 3
        assert float(r["best_reward_in_trajectory"]) == 0.75
        assert float(r["policy_loss"]) == 1.5
        assert r["running_best_formula"] == "A B <"
        assert r["strategy"] == "PUCT+Max"
        assert r["activity"] == "sleep"


def test_minus_inf_best_reward_writes_empty_string():
    """``-inf`` must render as '' so pandas.to_numeric parses cleanly."""
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep")
        logger.log_iteration(
            iteration=0,
            trajectory_length=0,
            best_reward_in_trajectory=float("-inf"),
            n_failed_evaluations=5,
            policy_loss=0.0,
            value_loss=0.0,
            total_loss=0.0,
        )
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["best_reward_in_trajectory"] == ""
        assert rows[0]["running_best_reward"] == ""


def test_running_best_tracks_max_and_holds_formula():
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep")
        logger.log_iteration(
            iteration=0, trajectory_length=1,
            best_reward_in_trajectory=0.2, n_failed_evaluations=0,
            policy_loss=0, value_loss=0, total_loss=0,
            best_formula="A",
        )
        logger.log_iteration(
            iteration=1, trajectory_length=1,
            best_reward_in_trajectory=0.9, n_failed_evaluations=0,
            policy_loss=0, value_loss=0, total_loss=0,
            best_formula="A B <",
        )
        logger.log_iteration(
            iteration=2, trajectory_length=1,
            best_reward_in_trajectory=0.5, n_failed_evaluations=0,
            policy_loss=0, value_loss=0, total_loss=0,
            best_formula="C",
        )
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        # Row 2: best was 0.9 from iter 1
        assert float(rows[-1]["running_best_reward"]) == 0.9
        assert rows[-1]["running_best_formula"] == "A B <"


def test_auto_numbered_run_indices():
    with tempfile.TemporaryDirectory() as td:
        a = AlphaZeroCSVLogger(td, "ENV-v0_sleep", "sleep", strategy="PUCT+Max")
        b = AlphaZeroCSVLogger(td, "ENV-v0_sleep", "sleep", strategy="PUCT+Max")
        c = AlphaZeroCSVLogger(td, "ENV-v0_sleep", "sleep", strategy="UCB1+Legacy")
        assert a.run == 0
        assert b.run == 1
        assert c.run == 0          # different strategy → fresh counter


# --------------------------------------------------------------------------- #
# log_training_log (convenience)
# --------------------------------------------------------------------------- #

def test_log_training_log_writes_every_iteration():
    log = TrainingLog(best_formula="A B <", best_reward=0.9)
    log.iterations.append(IterationLog(
        iteration=0, trajectory_length=3, best_reward_in_trajectory=0.9,
        n_failed_evaluations=1, train_total=1.0, train_policy=0.7, train_value=0.3,
    ))
    log.iterations.append(IterationLog(
        iteration=1, trajectory_length=3, best_reward_in_trajectory=0.6,
        n_failed_evaluations=0, train_total=0.5, train_policy=0.3, train_value=0.2,
    ))

    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep", strategy="PUCT+Max")
        logger.log_training_log(log)
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert float(rows[0]["best_reward_in_trajectory"]) == 0.9
        assert rows[0]["running_best_formula"] == "A B <"


def test_as_on_iteration_end_reads_best_formula_in_trajectory():
    """
    Regression: ``as_on_iteration_end()`` callback must populate the
    logger's running_best_formula from
    ``IterationLog.best_formula_in_trajectory`` when the caller didn't
    supply its own provider. Without this, ``train()``'s default
    callback left ``running_best_formula`` empty (pandas NaN) even
    though ``train()`` knew which rule earned the best reward.
    """
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep", strategy="PUCT+Max")
        cb = logger.as_on_iteration_end()

        cb(IterationLog(
            iteration=0, trajectory_length=2, best_reward_in_trajectory=0.4,
            n_failed_evaluations=0, train_total=0.1, train_policy=0.05, train_value=0.05,
            best_formula_in_trajectory="A",
        ))
        cb(IterationLog(
            iteration=1, trajectory_length=2, best_reward_in_trajectory=0.9,
            n_failed_evaluations=0, train_total=0.2, train_policy=0.1, train_value=0.1,
            best_formula_in_trajectory="A B <",
        ))
        cb(IterationLog(
            iteration=2, trajectory_length=2, best_reward_in_trajectory=0.5,
            n_failed_evaluations=0, train_total=0.1, train_policy=0.05, train_value=0.05,
            best_formula_in_trajectory="C",
        ))

        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))

        # Each iteration's running_best_formula reflects the best seen
        # so far. By iteration 2 the best ever is still "A B <" (0.9).
        assert rows[0]["running_best_formula"] == "A"
        assert rows[1]["running_best_formula"] == "A B <"
        assert rows[2]["running_best_formula"] == "A B <"    # not "C"
        # Reward tracker is monotonic.
        assert float(rows[2]["running_best_reward"]) == 0.9


def test_as_on_iteration_end_provider_overrides_iteration_log_field():
    """Explicit ``best_formula_provider`` wins over IterationLog field."""
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep")
        cb = logger.as_on_iteration_end(
            best_formula_provider=lambda _it: "PROVIDER_WINS",
        )
        cb(IterationLog(
            iteration=0, trajectory_length=1, best_reward_in_trajectory=1.0,
            n_failed_evaluations=0, train_total=0, train_policy=0, train_value=0,
            best_formula_in_trajectory="FIELD_LOSES",
        ))
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["running_best_formula"] == "PROVIDER_WINS"


# --------------------------------------------------------------------------- #
# v0.2.7 additions: eval + phase timings
# --------------------------------------------------------------------------- #

def test_new_csv_columns_in_header():
    """Header includes the 7 new columns added in v0.2.7."""
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep")
        with open(logger.csv_path) as f:
            header = next(csv.reader(f))
        for col in (
            "eval_reward", "eval_success_rate", "eval_episode_length",
            "t_mcts_s", "t_nn_train_s", "t_eval_s", "t_buffer_s",
        ):
            assert col in header, f"missing column {col!r} in {header}"


def test_log_iteration_writes_eval_and_timing_fields():
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep")
        logger.log_iteration(
            iteration=0, trajectory_length=1,
            best_reward_in_trajectory=1.0, n_failed_evaluations=0,
            policy_loss=0.1, value_loss=0.2, total_loss=0.3,
            eval_reward=5.5, eval_success_rate=0.75, eval_episode_length=12.0,
            t_mcts_s=0.5, t_nn_train_s=0.1, t_eval_s=0.05, t_buffer_s=0.01,
        )
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        r = rows[0]
        assert float(r["eval_reward"]) == 5.5
        assert float(r["eval_success_rate"]) == 0.75
        assert float(r["eval_episode_length"]) == 12.0
        assert float(r["t_mcts_s"]) == 0.5
        assert float(r["t_nn_train_s"]) == 0.1
        assert float(r["t_eval_s"]) == 0.05
        assert float(r["t_buffer_s"]) == 0.01


def test_log_iteration_empty_cells_when_eval_is_none():
    """Iterations where eval didn't fire write empty cells (-> NaN on read)."""
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep")
        logger.log_iteration(
            iteration=0, trajectory_length=1,
            best_reward_in_trajectory=1.0, n_failed_evaluations=0,
            policy_loss=0, value_loss=0, total_loss=0,
            # eval fields omitted — defaults to None
            t_mcts_s=0.2,
        )
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        r = rows[0]
        assert r["eval_reward"] == ""
        assert r["eval_success_rate"] == ""
        assert r["eval_episode_length"] == ""
        assert float(r["t_mcts_s"]) == 0.2
        # t_eval_s defaulted to 0.0 and rounds to an explicit "0.0" string.
        assert float(r["t_eval_s"]) == 0.0


def test_as_on_iteration_end_threads_eval_and_timings():
    """Callback reads the v0.2.7 IterationLog fields and writes them."""
    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(td, "ENV", "sleep")
        cb = logger.as_on_iteration_end()
        cb(IterationLog(
            iteration=0, trajectory_length=2,
            best_reward_in_trajectory=0.4, n_failed_evaluations=0,
            train_total=0.1, train_policy=0.05, train_value=0.05,
            best_formula_in_trajectory="A",
            eval_reward=5.5, eval_success_rate=0.75, eval_episode_length=12.0,
            t_mcts_s=0.5, t_nn_train_s=0.1, t_eval_s=0.05, t_buffer_s=0.01,
        ))
        with open(logger.csv_path) as f:
            rows = list(csv.DictReader(f))
        r = rows[0]
        assert float(r["eval_reward"]) == 5.5
        assert float(r["t_mcts_s"]) == 0.5
        assert r["running_best_formula"] == "A"


# --------------------------------------------------------------------------- #
# load_alphazero_logs
# --------------------------------------------------------------------------- #

def test_load_alphazero_logs_returns_empty_on_empty_dir():
    with tempfile.TemporaryDirectory() as td:
        df = load_alphazero_logs(td)
        # pandas import happens inside; we just check it returns a DataFrame
        assert len(df) == 0


def test_load_alphazero_logs_merges_files_and_config():
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        la = AlphaZeroCSVLogger(
            td, "ENV-v0_sleep", "sleep", strategy="PUCT+Max",
            config={"n_simulations": 50},
        )
        la.log_iteration(
            iteration=0, trajectory_length=1,
            best_reward_in_trajectory=0.5, n_failed_evaluations=0,
            policy_loss=0, value_loss=0, total_loss=0,
        )
        lb = AlphaZeroCSVLogger(
            td, "ENV-v0_sleep", "sleep", strategy="UCB1+Legacy",
            config={"n_simulations": 25},
        )
        lb.log_iteration(
            iteration=0, trajectory_length=2,
            best_reward_in_trajectory=0.2, n_failed_evaluations=1,
            policy_loss=0, value_loss=0, total_loss=0,
        )

        df = load_alphazero_logs(td)
        assert len(df) == 2
        strategies = set(df["strategy"].unique())
        assert strategies == {"PUCT+Max", "UCB1+Legacy"}
        assert "config_n_simulations" in df.columns
        # Row-level config values match the file's config
        az_row = df[df["strategy"] == "PUCT+Max"].iloc[0]
        assert az_row["config_n_simulations"] == 50
        assert "label" in df.columns
        assert "sleep" in str(az_row["label"])


def test_load_alphazero_logs_handles_list_and_dict_config_values():
    """
    Regression: a config with list/dict values (e.g. empty
    ``context_rules: []`` on the first box) previously broke
    ``load_alphazero_logs`` with::

        ValueError: Length of values (0) does not match length of index (20)

    because pandas tried to broadcast the 0-length list across the
    DataFrame's rows. Fixed by serialising non-scalar config values
    to JSON strings before the per-column assignment.
    """
    import json as _json

    with tempfile.TemporaryDirectory() as td:
        logger = AlphaZeroCSVLogger(
            td, "ENV-v0_bathe", "bathe", strategy="PUCT+Max",
            config={
                "n_simulations": 50,
                "context_rules": [],                          # length-0 list
                "event_types": ["A", "B", "C"],               # non-empty list
                "nested": {"c_puct": 1.5, "temperature": 1.0},# dict
            },
        )
        # Log more rows than the 0-length list has entries to expose
        # any broadcast-length mismatch.
        for i in range(5):
            logger.log_iteration(
                iteration=i, trajectory_length=1,
                best_reward_in_trajectory=0.1 * i,
                n_failed_evaluations=0,
                policy_loss=0, value_loss=0, total_loss=0,
            )

        df = load_alphazero_logs(td)
        assert len(df) == 5                                   # no crash

        # List/dict serialised to JSON string and broadcast to every row.
        assert df["config_context_rules"].iloc[0] == "[]"
        assert _json.loads(df["config_event_types"].iloc[0]) == ["A", "B", "C"]
        assert _json.loads(df["config_nested"].iloc[0]) == {
            "c_puct": 1.5, "temperature": 1.0,
        }
        # Scalar config values still stored as scalars.
        assert df["config_n_simulations"].iloc[0] == 50


def test_load_alphazero_logs_walks_nested_directories():
    """
    New nested layout ``logs/<strategy>/run_eval_<N>/*.csv`` must be
    discovered by the loader. Writes one CSV in each of two strategy
    subdirs and confirms both rows come back with the correct
    ``strategy`` column.
    """
    with tempfile.TemporaryDirectory() as td:
        strat_a_dir = os.path.join(td, "PUCT+Max", "run_eval_0")
        strat_b_dir = os.path.join(td, "PUCT+Max-GT", "run_eval_0")
        os.makedirs(strat_a_dir); os.makedirs(strat_b_dir)

        la = AlphaZeroCSVLogger(
            strat_a_dir, "ENV-v0_sleep", "sleep", strategy="PUCT+Max",
            config={"n_simulations": 50},
        )
        la.log_iteration(
            iteration=0, trajectory_length=1,
            best_reward_in_trajectory=0.5, n_failed_evaluations=0,
            policy_loss=0, value_loss=0, total_loss=0,
        )
        lb = AlphaZeroCSVLogger(
            strat_b_dir, "ENV-v0_sleep", "sleep", strategy="PUCT+Max-GT",
            config={"n_simulations": 25},
        )
        lb.log_iteration(
            iteration=0, trajectory_length=2,
            best_reward_in_trajectory=0.2, n_failed_evaluations=0,
            policy_loss=0, value_loss=0, total_loss=0,
        )

        df = load_alphazero_logs(td)
        assert len(df) == 2
        assert set(df["strategy"].unique()) == {"PUCT+Max", "PUCT+Max-GT"}
        # Both configs were merged from the per-CSV directory, not the
        # top-level ``td``.
        a_row = df[df["strategy"] == "PUCT+Max"].iloc[0]
        b_row = df[df["strategy"] == "PUCT+Max-GT"].iloc[0]
        assert a_row["config_n_simulations"] == 50
        assert b_row["config_n_simulations"] == 25
