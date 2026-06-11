"""
Tests for ``train(..., device=...)`` (v0.2.7+).

Pins:
    - ``device=None`` resolves to ``"cpu"`` on a CUDA-less install (what
      the CI / this dev box runs) and to ``"cuda"`` when available.
    - Explicit ``device="cpu"`` always works; the resolved device is
      recorded on ``TrainingLog.device``.
    - Asking for CUDA on a CPU-only torch build raises a clear error
      instead of silently training on CPU.
    - The model's parameters end up on the resolved device (checked
      indirectly by running one iteration and re-reading the model
      from the captured closure — here we just pin that CPU keeps
      training working with the new code path).
"""
from __future__ import annotations

import pytest
import torch

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.training.train import train


class _ConstSim:
    def evaluate(self, node):
        return 1.0


def _tiny(**kwargs):
    return train(
        grammar=AllenIntervalGrammar(event_types=("A", "B"), relations=("<",)),
        expensive_simulator=_ConstSim(),
        n_iterations=2,
        n_simulations=4,
        depth_limit=2,
        seed=0,
        buffer_warmup=1,
        train_steps_per_iteration=1,
        max_len=12,
        d_model=16,
        nhead=2,
        num_layers=1,
        **kwargs,
    )


def test_device_auto_detects_best_available():
    """
    With ``device=None``, ``train()`` must resolve to ``"cuda"`` when
    ``torch.cuda.is_available()``, otherwise ``"cpu"``. Training itself
    must work end-to-end on whichever was picked.
    """
    log = _tiny()
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.device stringifies "cuda" as "cuda" (without index) when no
    # index is specified; accept both forms.
    assert log.device.startswith(expected)
    assert len(log.iterations) == 2


def test_device_explicit_cpu():
    log = _tiny(device="cpu")
    assert log.device == "cpu"
    assert len(log.iterations) == 2


def test_device_cuda_without_cuda_build_raises():
    """
    If a caller asks for CUDA on a CPU-only torch install, ``train()``
    must raise a clear RuntimeError rather than silently falling back
    or crashing deep inside ``.to()``.
    """
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine; the error path is unreachable.")
    with pytest.raises(RuntimeError, match="CPU-only build"):
        _tiny(device="cuda")


def test_device_cuda_runs_when_available():
    """Smoke test: explicitly request CUDA and run a couple of iterations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this torch install.")
    log = _tiny(device="cuda")
    assert log.device.startswith("cuda")
    assert len(log.iterations) == 2
