"""
Tokeniser that turns rule strings into integer tensors.

The vocabulary is built from a ``Grammar`` instance plus four special
tokens: ``<PAD>``, ``<BOS>``, ``<EOS>``, and ``<END>`` (the rule
terminator). Every token a grammar exposes gets exactly one id.
``encode`` produces a fixed-length ``LongTensor`` padded to ``max_len``
with the PAD id; ``decode`` is provided for round-trip tests.

Lazy torch import: importing this module does NOT import ``torch``
unless ``encode`` is called. That lets the rest of the package (grammar,
mcts, replay) be used without the NN dependency.
"""
from __future__ import annotations

from typing import List

from alpha_rule.grammar.grammar import Grammar


PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
END_MARKER = "<END>"  # the grammar's ``apply`` appends this when END_RULE
                      # finishes a rule ("A B < <END>"). Distinct from EOS,
                      # which marks the end of the tokenised sequence.
SPECIAL_TOKENS = (PAD, BOS, EOS, END_MARKER)


class GrammarTokenizer:
    """
    Vocab-aware tokeniser for grammar-derived rule strings.

    Args:
        grammar: source of grammar tokens (event types, relations,
            ``END_RULE``).

    Attributes:
        vocab: ordered list, indices are stable.
        id_of: name -> id map.
        pad_id, bos_id, eos_id: cached special-token ids.
    """

    PAD = PAD
    BOS = BOS
    EOS = EOS

    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.vocab: List[str] = list(SPECIAL_TOKENS) + list(grammar.vocab())
        # Defensive: enforce uniqueness so id_of is well-defined.
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError(
                f"Grammar vocab collides with special tokens: {self.vocab}"
            )
        self.id_of = {tok: i for i, tok in enumerate(self.vocab)}
        self.token_of = {i: tok for tok, i in self.id_of.items()}

        self.pad_id = self.id_of[PAD]
        self.bos_id = self.id_of[BOS]
        self.eos_id = self.id_of[EOS]

    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------ #
    # encode / decode
    # ------------------------------------------------------------------ #

    def encode(self, rule_string: str, max_len: int):
        """
        Tokenise ``rule_string``, prepend BOS, append EOS, pad to
        ``max_len`` with PAD. Returns a 1-D ``torch.LongTensor`` of
        length ``max_len`` (truncated from the right if too long).
        """
        import torch  # lazy

        tokens = rule_string.split() if rule_string else []
        # Replace the empty-root sentinel "<ROOT>" with no tokens so
        # the encoded sequence is just BOS + EOS + padding.
        tokens = [t for t in tokens if t != "<ROOT>"]

        ids = [self.bos_id] + [self._lookup(t) for t in tokens] + [self.eos_id]
        if len(ids) > max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        """Round-trip helper. Strips PAD/BOS/EOS, joins with spaces."""
        out = []
        for i in ids:
            tok = self.token_of[int(i)]
            if tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return " ".join(out)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _lookup(self, token: str) -> int:
        if token not in self.id_of:
            raise KeyError(
                f"Token {token!r} not in tokenizer vocab. "
                f"Either the grammar omits it or you're decoding a "
                f"pre-grammar rule string."
            )
        return self.id_of[token]
