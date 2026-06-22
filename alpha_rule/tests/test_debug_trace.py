"""
Tests for the debug-branch Q-trace + decision-trace collectors
(``alpha_rule.mcts.debug_trace``). These guard the instrumentation used to
diagnose the policy-collapse failure; they are debug-only and may be removed with
the rest of the seam.
"""
from __future__ import annotations

from alpha_rule.mcts.debug_trace import DecisionTraceCollector, QTraceCollector


class _Node:
    def __init__(self, name, *, level=0, parent=None, N=1, Q_max=0.0, Q_sum=0.0,
                 N_passers=0, prior=1.0, parent_action=None, is_dead=False,
                 children=None):
        self.name = name
        self.level = level
        self.parent = parent
        self.N = N
        self.Q_max = Q_max
        self.Q_sum = Q_sum
        self.N_passers = N_passers
        self.prior = prior
        self.parent_action = parent_action
        self.is_dead = is_dead
        self.children = children or []


def test_record_eval_walks_to_root_and_tags_source():
    root = _Node("R", level=0, N=10, Q_max=2.0)
    a = _Node("A", level=1, parent=root, N=6, Q_max=2.0)
    leaf = _Node("A A", level=2, parent=a, N=3, Q_max=1.0)

    qc = QTraceCollector()
    qc.record_eval(leaf, 1.0, "nn")
    qc.record_eval(leaf, -0.5, "sim")

    # The leaf and both ancestors each received both values.
    for name in ("A A", "A", "R"):
        assert len(qc._iter[name]["entries"]) == 2

    # is_direct only on the evaluated leaf; ancestors are backprop.
    assert all(d for _, _, d in qc._iter["A A"]["entries"])
    assert all(not d for _, _, d in qc._iter["A"]["entries"])

    # Source is preserved and bucketable.
    assert sorted(s for _, s, _ in qc._iter["R"]["entries"]) == ["nn", "sim"]


def test_report_rolls_into_store_and_clears_buffer(capsys):
    root = _Node("R", level=0, N=4, Q_max=1.0, Q_sum=2.0, N_passers=2)
    leaf = _Node("R x", level=1, parent=root, N=2, Q_max=1.0, Q_sum=2.0, N_passers=2)
    qc = QTraceCollector()
    qc.record_eval(leaf, 1.0, "sim")
    qc.record_eval(leaf, 0.5, "nn")

    qc.report(iteration=0)
    out = capsys.readouterr().out
    assert "level 0" in out and "level 1" in out and "sim" in out and "nn" in out

    assert qc._iter == {}                       # per-iteration buffer cleared
    assert 0 in qc._all and "R x" in qc._all[0]  # snapshot kept for the dump
    assert len(qc._all[0]["R x"]["entries"]) == 2


def test_decision_trace_captures_children_and_commit():
    c1 = _Node("A A", parent_action="A", prior=0.4, Q_sum=2.0, N_passers=4,
               Q_max=1.0, N=10)
    c2 = _Node("A C", parent_action="C", prior=0.1, Q_sum=3.0, N_passers=4,
               Q_max=1.1, N=5)
    parent = _Node("A", children=[c1, c2])

    dc = DecisionTraceCollector()
    dc.record_decision(iteration=0, depth_step=1, parent=parent, committed_action="A")

    rec = dc.records[0]
    assert rec["node"] == "A" and rec["committed_action"] == "A"
    assert {ch["action"] for ch in rec["children"]} == {"A", "C"}
    a_child = next(ch for ch in rec["children"] if ch["action"] == "A")
    assert a_child["prior"] == 0.4 and a_child["q_fmean"] == 0.5 and a_child["N"] == 10
