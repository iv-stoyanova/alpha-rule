"""
DEBUG-ONLY instrumentation (lives on the ``debug`` branch; safe to delete).

Two collectors used to diagnose the "policy commits the high-prior branch and
ignores a marginally-better-deeper one" failure:

    QTraceCollector       records every node EVALUATION's value tagged by source
                          ("sim" = real simulator, "nn" = value head), and
                          propagates it to all ancestors (so each node carries
                          every value that fed its Q). Per iteration it prints the
                          top-K nodes of each level (ranked by Q_max) with the
                          overall and per-source value distributions, then keeps
                          the per-iteration data for a JSON dump.

    DecisionTraceCollector records, at each committed construction step, every
                          child's prior / Q / visit count and which child was
                          committed -- the structured data the H1-H6 tests need.

Everything is opt-in: ``train(debug_trace_dir=...)`` builds these and threads them
through ``run_self_play``; with no dir nothing is constructed and the run is
byte-identical.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def _finite(values: List[float]) -> List[float]:
    return [v for v in values if math.isfinite(v)]


def _stats(values: List[float]) -> Optional[dict]:
    """count / mean / std / 5-number percentiles over the FINITE values, plus a
    count of non-finite (-inf) entries. None if no finite values."""
    n_total = len(values)
    fin = _finite(values)
    n_neg_inf = sum(1 for v in values if v == float("-inf"))
    if not fin:
        return {"n": n_total, "n_finite": 0, "n_neg_inf": n_neg_inf}
    arr = np.asarray(fin, dtype=float)
    pct = np.percentile(arr, [0, 25, 50, 75, 100])
    return {
        "n": n_total,
        "n_finite": len(fin),
        "n_neg_inf": n_neg_inf,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(pct[0]), "p25": float(pct[1]), "median": float(pct[2]),
        "p75": float(pct[3]), "max": float(pct[4]),
    }


def _fmt(s: Optional[dict]) -> str:
    if not s or s.get("n_finite", 0) == 0:
        return f"n={s['n'] if s else 0} (no finite values)"
    return (f"n={s['n']} mean={s['mean']:+.2f} std={s['std']:.2f} "
            f"[{s['min']:+.2f} {s['p25']:+.2f} {s['median']:+.2f} "
            f"{s['p75']:+.2f} {s['max']:+.2f}]"
            + (f" (-inf x{s['n_neg_inf']})" if s["n_neg_inf"] else ""))


class QTraceCollector:
    """Per-node value tracing tagged by source, with a per-iteration report."""

    def __init__(self):
        # Per-iteration working buffer: name -> {"node", "level", "entries"}.
        self._iter: Dict[str, dict] = {}
        # Cross-iteration store for the JSON dump: iteration -> name -> record.
        self._all: Dict[int, dict] = {}

    def record_eval(self, node, value: float, source: str) -> None:
        """Record one leaf evaluation and propagate it up the ancestor chain
        (mirroring the backup walk), tagging each contribution with its source
        and whether the recorded node is the one that was evaluated (direct)."""
        anc = node
        while anc is not None:
            e = self._iter.get(anc.name)
            if e is None:
                e = {"node": anc, "level": anc.level, "entries": []}
                self._iter[anc.name] = e
            elif anc.N > e["node"].N:          # keep the most-developed node ref
                e["node"] = anc
            e["entries"].append((float(value), source, anc is node))
            anc = anc.parent

    def report(self, iteration: int, top_k: int = 5) -> None:
        """Print the top-K nodes of each level (ranked by Q_max) with overall and
        per-source value distributions, then roll this iteration into the store."""
        if not self._iter:
            return
        by_level: Dict[int, list] = {}
        for name, e in self._iter.items():
            by_level.setdefault(e["level"], []).append((name, e))

        print(f"  [it={iteration}] Q-trace: top-{top_k} nodes/level "
              f"(ranked by Q_max; dist = n mean std [min p25 med p75 max])")
        for level in sorted(by_level):
            nodes = by_level[level]
            nodes.sort(key=lambda kv: (kv[1]["node"].Q_max
                                       if math.isfinite(kv[1]["node"].Q_max)
                                       else -1e18), reverse=True)
            print(f"    level {level}:")
            for name, e in nodes[:top_k]:
                node = e["node"]
                vals = [v for v, _, _ in e["entries"]]
                sim = [v for v, s, _ in e["entries"] if s == "sim"]
                nn = [v for v, s, _ in e["entries"] if s == "nn"]
                n_direct = sum(1 for _, _, d in e["entries"] if d)
                n_bp = len(e["entries"]) - n_direct
                qfm = (node.Q_sum / node.N_passers) if node.N_passers > 0 else float("nan")
                print(f"      {name!r}  N={node.N} Q_max={node.Q_max:+.2f} "
                      f"Q_fmean={qfm:+.2f}  (direct={n_direct} backprop={n_bp})")
                print(f"         all: {_fmt(_stats(vals))}")
                print(f"         sim: {_fmt(_stats(sim))}")
                print(f"         nn : {_fmt(_stats(nn))}")

        # Roll into the cross-iteration store and clear the working buffer.
        snap = {}
        for name, e in self._iter.items():
            node = e["node"]
            snap[name] = {
                "level": e["level"],
                "N": node.N,
                "q_max": node.Q_max if math.isfinite(node.Q_max) else None,
                "q_fmean": (node.Q_sum / node.N_passers) if node.N_passers > 0 else None,
                "entries": [[v, s, bool(d)] for v, s, d in e["entries"]],
            }
        self._all[iteration] = snap
        self._iter = {}

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._all, f)


class DecisionTraceCollector:
    """Per committed step: every child's prior/Q/N + the committed action."""

    def __init__(self):
        self.records: List[dict] = []

    def record_decision(self, iteration: int, depth_step: int, parent,
                        committed_action: str) -> None:
        children = []
        for c in parent.children:
            qfm = (c.Q_sum / c.N_passers) if c.N_passers > 0 else None
            children.append({
                "action": c.parent_action,
                "prior": float(c.prior),
                "q_fmean": qfm,
                "q_max": c.Q_max if math.isfinite(c.Q_max) else None,
                "N": c.N,
                "is_dead": bool(c.is_dead),
            })
        self.records.append({
            "iteration": iteration,
            "depth_step": depth_step,
            "node": parent.name,
            "committed_action": committed_action,
            "children": children,
        })

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f)
