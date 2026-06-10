"""
Tests for ``nn.tokenizer.GrammarTokenizer``.

Pins:
    - Vocab size = 3 specials + grammar.vocab() length.
    - ``encode`` produces a fixed-length 1-D LongTensor padded with PAD.
    - ``encode`` strips the ``<ROOT>`` sentinel so the network sees
      only "real" tokens.
    - ``decode(encode(s))`` round-trip recovers the rule string
      (modulo whitespace normalisation).
    - Padding works at the cap.
    - Truncation triggers when the sequence is longer than ``max_len``.
"""
from __future__ import annotations

import torch

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.nn.tokenizer import BOS, EOS, PAD, GrammarTokenizer


def _tokenizer():
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<", ">"))
    return GrammarTokenizer(g)


def test_vocab_size_is_specials_plus_grammar():
    tok = _tokenizer()
    # specials = PAD, BOS, EOS, END_MARKER = 4
    expected = 4 + 2 + 2 + 1                 # specials + 2 events + 2 rels + END_RULE
    assert tok.vocab_size() == expected


def test_special_token_ids_unique():
    tok = _tokenizer()
    assert tok.pad_id != tok.bos_id != tok.eos_id


def test_encode_returns_long_tensor_of_max_len():
    tok = _tokenizer()
    out = tok.encode("A B <", max_len=10)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.long
    assert out.shape == (10,)


def test_encode_pads_with_pad_id():
    tok = _tokenizer()
    out = tok.encode("A", max_len=8).tolist()
    # BOS, A, EOS, then PAD * 5
    assert out[0] == tok.bos_id
    assert out[1] == tok.id_of["A"]
    assert out[2] == tok.eos_id
    assert all(x == tok.pad_id for x in out[3:])


def test_encode_strips_root_sentinel():
    tok = _tokenizer()
    out = tok.encode("<ROOT>", max_len=4).tolist()
    # BOS, EOS, PAD, PAD
    assert out[0] == tok.bos_id
    assert out[1] == tok.eos_id
    assert out[2] == tok.pad_id


def test_decode_round_trip_strips_specials():
    tok = _tokenizer()
    encoded = tok.encode("A B <", max_len=10)
    assert tok.decode(encoded) == "A B <"


def test_encode_truncates_when_too_long():
    tok = _tokenizer()
    # Force a short cap so truncation kicks in.
    out = tok.encode("A B < > <", max_len=4)
    assert out.shape == (4,)


def test_unknown_token_raises():
    tok = _tokenizer()
    try:
        tok.encode("Q", max_len=5)
    except KeyError as e:
        assert "Q" in str(e)
    else:
        raise AssertionError("expected KeyError for unknown token")
