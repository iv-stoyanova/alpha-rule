"""
Tests for ``nn.model.AllenFormulaNet``.

Pins:
    - Forward pass produces ``(policy_logits, value)`` of expected
      shapes.
    - CPU-only and deterministic given a seed.
    - The encoder respects ``max_len`` (rejects too-long inputs).
"""
from __future__ import annotations

import torch

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.nn.model import AllenFormulaNet
from alpha_rule.nn.tokenizer import GrammarTokenizer


def _model(d_model=16, num_layers=1, max_len=12):
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(
        tok, d_model=d_model, nhead=2, num_layers=num_layers, max_len=max_len,
    )
    return model, tok


def test_forward_shapes():
    model, tok = _model()
    batch = torch.stack([
        tok.encode("A", max_len=12),
        tok.encode("A B <", max_len=12),
    ])                                      # (2, 12)
    logits, value = model(batch)
    assert logits.shape == (2, tok.vocab_size())
    assert value.shape == (2,)


def test_deterministic_with_seed():
    torch.manual_seed(0)
    model_a, tok = _model()
    torch.manual_seed(0)
    model_b, _ = _model()

    batch = tok.encode("A B <", max_len=12).unsqueeze(0)
    logits_a, value_a = model_a(batch)
    logits_b, value_b = model_b(batch)
    assert torch.allclose(logits_a, logits_b)
    assert torch.allclose(value_a, value_b)


def test_input_length_must_not_exceed_max_len():
    model, tok = _model(max_len=4)
    too_long = torch.zeros((1, 5), dtype=torch.long)
    try:
        model(too_long)
    except ValueError as e:
        assert "max_len" in str(e)
    else:
        raise AssertionError("expected ValueError for over-length input")


def test_pooler_module_present_and_changes_output():
    """
    Pin: the encoder applies a learned (Linear + Tanh) projection on top
    of the BOS-position hidden state before returning. Swapping the
    pooler for ``nn.Identity`` must change the output, proving the
    projection is actually wired in (not optimised away).
    """
    from torch import nn as _nn

    torch.manual_seed(0)
    model, tok = _model()
    batch = torch.stack([
        tok.encode("A", max_len=12),
        tok.encode("A B <", max_len=12),
    ])

    # Output WITH the learned pooler.
    with torch.no_grad():
        pooled_with = model.encoder(batch).clone()

    # Pooler must exist as an attribute on the encoder.
    assert hasattr(model.encoder, "pooler"), \
        "encoder must expose a `pooler` attribute (CLS-style learned projection)"

    # Output WITHOUT the pooler (Identity replacement).
    original_pooler = model.encoder.pooler
    model.encoder.pooler = _nn.Identity()
    try:
        with torch.no_grad():
            pooled_without = model.encoder(batch).clone()
    finally:
        model.encoder.pooler = original_pooler

    assert not torch.allclose(pooled_with, pooled_without), \
        "pooler must change the encoder output (Linear+Tanh on BOS hidden state)"
