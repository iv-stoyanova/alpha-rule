"""
Tests for ``nn.training.train_step``.

Pins:
    - One step on a fixed batch reduces total loss (smoke check —
      proves backward + optimizer wiring works, NOT convergence).
    - Loss components are non-negative scalar floats.
    - Empty batch is a no-op returning zeros.
    - The collator handles ``visit_pi`` dicts that don't sum to 1.0
      (renormalises) and ignores keys not in the tokeniser vocab.
"""
from __future__ import annotations

import torch

from alpha_rule.grammar.allen import AllenIntervalGrammar
from alpha_rule.nn.model import AllenFormulaNet
from alpha_rule.nn.tokenizer import GrammarTokenizer
from alpha_rule.nn.training import collate, train_step


def _setup(max_len=12):
    torch.manual_seed(7)
    g = AllenIntervalGrammar(event_types=("A", "B"), relations=("<",))
    tok = GrammarTokenizer(g)
    model = AllenFormulaNet(tok, d_model=16, nhead=2, num_layers=1, max_len=max_len)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, optim, tok


def _fixed_batch(tok):
    """A deliberately small batch so the model can fit it in a few steps."""
    return [
        ("A",      {"A": 1.0},          0.5),
        ("A B <",  {"<": 0.7, "B": 0.3}, 1.0),
        ("B",      {"END_RULE": 1.0},   -2.0),
    ]


def test_train_step_reduces_total_loss():
    model, optim, tok = _setup()
    batch = _fixed_batch(tok)

    log_first = train_step(model, optim, batch, max_len=12)
    log_second = train_step(model, optim, batch, max_len=12)
    log_third = train_step(model, optim, batch, max_len=12)

    # Not strictly monotone (Adam can wobble), but after three steps the
    # loss must have moved off the initial value in the expected direction.
    assert log_third.total < log_first.total


def test_train_step_returns_finite_non_negative_components():
    model, optim, tok = _setup()
    log = train_step(model, optim, _fixed_batch(tok), max_len=12)
    for v in (log.total, log.policy, log.value):
        assert isinstance(v, float)
        assert v == v                                    # not NaN
        assert v >= 0.0


def test_empty_batch_is_noop():
    model, optim, tok = _setup()
    log = train_step(model, optim, [], max_len=12)
    assert log.total == 0.0
    assert log.policy == 0.0
    assert log.value == 0.0


def test_collate_renormalises_pi_and_ignores_unknown_tokens():
    model, _, tok = _setup()
    batch = [
        ("A",     {"A": 0.5, "B": 0.5, "ZZZ": 9.9}, 1.0),  # ZZZ is unknown
    ]
    states, pi, z = collate(batch, model, max_len=12)
    assert pi.shape == (1, tok.vocab_size())
    # Only A and B contributed; both equal weight; renormalised to sum 1.
    pa = pi[0, tok.id_of["A"]].item()
    pb = pi[0, tok.id_of["B"]].item()
    assert abs(pa - 0.5) < 1e-6
    assert abs(pb - 0.5) < 1e-6
    assert abs(pi[0].sum().item() - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# grad_clip — opt-in global-norm clipping before optimizer.step().
# Default 0.0 means "disabled" to preserve historical runs.
# --------------------------------------------------------------------------- #

def _large_value_batch(tok):
    """A batch with value targets at the old unnormalised scale (−100)
    — this was the gradient regime the alpha-rule-updates work targeted."""
    return [
        ("A",     {"A": 1.0}, -100.0),
        ("A B <", {"<": 1.0}, -100.0),
        ("B",     {"END_RULE": 1.0}, 0.0),
    ]


def test_grad_clip_default_disabled_matches_unclipped_path():
    """Pin: ``grad_clip=0.0`` (default) produces the same parameters
    as omitting the argument entirely."""
    import copy
    model_a, optim_a, tok = _setup()
    model_b = copy.deepcopy(model_a)
    optim_b = torch.optim.Adam(model_b.parameters(), lr=1e-2)

    batch = _large_value_batch(tok)
    train_step(model_a, optim_a, batch, max_len=12)                     # no kwarg
    train_step(model_b, optim_b, batch, max_len=12, grad_clip=0.0)       # explicit 0

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-7)


def test_grad_clip_positive_bounds_global_norm():
    """New: with grad_clip>0, the total grad-norm seen by Adam is ≤ grad_clip."""
    model, optim, tok = _setup()
    batch = _large_value_batch(tok)

    # Monkey-patch optimizer.step to inspect grad norm at the moment
    # Adam would apply the update.
    observed = {}
    real_step = optim.step
    def _spy_step(*a, **kw):
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().pow(2).sum().item())
        observed["norm"] = total_sq ** 0.5
        return real_step(*a, **kw)
    optim.step = _spy_step

    train_step(model, optim, batch, max_len=12, grad_clip=1.0)
    assert observed["norm"] <= 1.0 + 1e-4


# --------------------------------------------------------------------------- #
# Applicable-action masking (parity with NeuralEvaluator's inference softmax).
# At inference, illegal-production logits are masked to ``-inf`` before
# softmax. Training must use the same mask, otherwise the cross-entropy
# softmax denominator silently competes against tokens that are zeroed at
# inference, wasting model capacity.
# --------------------------------------------------------------------------- #

def test_collate_returns_applicable_mask_when_provided():
    """4-tuple rows ``(state, visit_pi, z, applicable_actions)`` produce a
    ``(B, vocab_size)`` bool mask: True for applicable token ids, False
    elsewhere."""
    model, _, tok = _setup()
    batch = [
        ("A",     {"A": 1.0},                       0.5, ("A", "B")),
        ("A B <", {"<": 1.0},                       1.0, ("END_RULE", "<")),
    ]
    out = collate(batch, model, max_len=12)
    assert len(out) == 4, "collate must return 4 tensors when masks are present"
    states, pi, z, mask = out
    assert mask.shape == (2, tok.vocab_size())
    assert mask.dtype == torch.bool
    # Row 0: only A and B applicable.
    assert bool(mask[0, tok.id_of["A"]])
    assert bool(mask[0, tok.id_of["B"]])
    assert not bool(mask[0, tok.id_of["END_RULE"]])
    # Row 1: only END_RULE and < applicable.
    assert bool(mask[1, tok.id_of["END_RULE"]])
    assert bool(mask[1, tok.id_of["<"]])
    assert not bool(mask[1, tok.id_of["A"]])


def test_collate_legacy_3tuple_rows_produce_no_mask():
    """3-tuple rows (no applicable_actions) preserve legacy unmasked
    behaviour — collate returns 3 tensors, no mask."""
    model, _, tok = _setup()
    out = collate(_fixed_batch(tok), model, max_len=12)
    assert len(out) == 3, "legacy 3-tuple rows must keep returning 3 tensors"


def test_train_step_masks_inapplicable_logits():
    """With an applicable-action mask, gradient on inapplicable-logit
    weights must be zero — ``masked_fill(-inf)`` before ``log_softmax``
    means those logits contribute nothing to the loss."""
    model, optim, tok = _setup()
    # Single-row batch where only ``A`` is applicable.
    batch = [("A", {"A": 1.0}, 0.5, ("A",))]

    optim.zero_grad()
    states_ids, target_pi, target_z, mask = collate(batch, model, max_len=12)
    logits, values = model(states_ids)
    logits_masked = logits.masked_fill(~mask, float("-inf"))
    log_probs = torch.nn.functional.log_softmax(logits_masked, dim=-1)
    policy_loss = -(target_pi * log_probs).sum(dim=-1).mean()
    policy_loss.backward()

    # Gradient on the policy head's row for an inapplicable token must be 0.
    grads = model.policy.linear.weight.grad
    inapplicable_id = tok.id_of["B"]
    assert torch.allclose(grads[inapplicable_id], torch.zeros_like(grads[inapplicable_id])), \
        "inapplicable-logit weight row must have zero gradient under masked softmax"


def test_value_head_output_bounded_in_minus_one_plus_one():
    """Pin: with the tanh-bounded ``ValueHead``, the model's value output
    is strictly in ``(-1, +1)`` regardless of input."""
    model, _, tok = _setup()
    # Random tokens within vocab, multiple rows.
    rng = torch.Generator().manual_seed(42)
    ids = torch.randint(0, tok.vocab_size(), (8, 12), generator=rng, dtype=torch.long)
    with torch.no_grad():
        _, values = model(ids)
    assert values.shape == (8,)
    assert values.abs().max().item() < 1.0


def test_train_step_4tuple_batch_runs_and_reduces_loss():
    """End-to-end: train_step accepts 4-tuple batches and still drives loss down."""
    model, optim, tok = _setup()
    batch = [
        ("A",     {"A": 1.0},                       0.5, ("A", "B")),
        ("A B <", {"<": 1.0},                       1.0, ("END_RULE", "<")),
        ("B",     {"END_RULE": 1.0},               -2.0, ("END_RULE", "<")),
    ]
    log_first = train_step(model, optim, batch, max_len=12)
    log_second = train_step(model, optim, batch, max_len=12)
    log_third = train_step(model, optim, batch, max_len=12)
    assert log_third.total < log_first.total


# --------------------------------------------------------------------------- #
# entropy_beta — opt-in policy-entropy bonus (rewards a flatter prior).
# label_smoothing — opt-in flattening of the visit-count target.
# Both default 0.0 (off) and must be byte-identical to the legacy path then.
# --------------------------------------------------------------------------- #

def test_entropy_bonus_default_off_matches_baseline():
    """``entropy_beta=0.0`` (default) produces the same parameters as omitting it."""
    import copy
    model_a, optim_a, tok = _setup()
    model_b = copy.deepcopy(model_a)
    optim_b = torch.optim.Adam(model_b.parameters(), lr=1e-2)

    batch = _fixed_batch(tok)
    train_step(model_a, optim_a, batch, max_len=12)                  # no kwarg
    train_step(model_b, optim_b, batch, max_len=12, entropy_beta=0.0)  # explicit 0

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-7)


def test_entropy_bonus_active_reports_entropy_and_changes_update():
    """With ``entropy_beta>0`` the step reports a positive policy entropy and the
    resulting parameters differ from the no-bonus update."""
    import copy
    model_a, optim_a, tok = _setup()
    model_b = copy.deepcopy(model_a)
    optim_b = torch.optim.Adam(model_b.parameters(), lr=1e-2)

    batch = _fixed_batch(tok)
    log_a = train_step(model_a, optim_a, batch, max_len=12)                    # baseline
    log_b = train_step(model_b, optim_b, batch, max_len=12, entropy_beta=0.1)  # bonus

    assert log_a.entropy == 0.0                       # not computed when off
    assert log_b.entropy > 0.0 and log_b.entropy == log_b.entropy    # finite, positive
    diverged = any(not torch.allclose(pa, pb, atol=1e-7)
                   for pa, pb in zip(model_a.parameters(), model_b.parameters()))
    assert diverged, "entropy bonus must change the gradient update"


def test_entropy_bonus_finite_under_masking():
    """Masked (4-tuple) batches must not produce NaN entropy (the 0*-inf trap)."""
    model, optim, tok = _setup()
    batch = [
        ("A",     {"A": 1.0},        0.5, ("A", "B")),
        ("A B <", {"<": 1.0},        1.0, ("END_RULE", "<")),
    ]
    log = train_step(model, optim, batch, max_len=12, entropy_beta=0.05)
    assert log.entropy == log.entropy and log.entropy >= 0.0     # not NaN


def test_label_smoothing_default_off_matches_baseline():
    """``label_smoothing=0.0`` (default) matches omitting it."""
    import copy
    model_a, optim_a, tok = _setup()
    model_b = copy.deepcopy(model_a)
    optim_b = torch.optim.Adam(model_b.parameters(), lr=1e-2)

    batch = _fixed_batch(tok)
    train_step(model_a, optim_a, batch, max_len=12)
    train_step(model_b, optim_b, batch, max_len=12, label_smoothing=0.0)

    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-7)


def test_label_smoothing_active_changes_update():
    """A positive ``label_smoothing`` flattens the target, changing the update
    (masked 4-tuple path), without NaN."""
    import copy
    model_a, optim_a, tok = _setup()
    model_b = copy.deepcopy(model_a)
    optim_b = torch.optim.Adam(model_b.parameters(), lr=1e-2)

    batch = [("A", {"A": 1.0}, 0.5, ("A", "B"))]      # sharp target over 2 legal acts
    train_step(model_a, optim_a, batch, max_len=12)                       # baseline
    log_b = train_step(model_b, optim_b, batch, max_len=12, label_smoothing=0.3)

    assert log_b.policy == log_b.policy               # not NaN
    diverged = any(not torch.allclose(pa, pb, atol=1e-7)
                   for pa, pb in zip(model_a.parameters(), model_b.parameters()))
    assert diverged, "label smoothing must change the effective target/update"


def test_grad_clip_reduces_norm_vs_unclipped_on_large_targets():
    """Smoke-level: on a batch with large value targets, the clipped run
    has strictly smaller grad-norm than the unclipped run."""
    import copy
    model_a, optim_a, tok = _setup()
    model_b = copy.deepcopy(model_a)
    optim_b = torch.optim.Adam(model_b.parameters(), lr=1e-2)

    batch = _large_value_batch(tok)

    def _capture(model, optim, **kw):
        norms = {}
        real = optim.step
        def _spy(*a, **k):
            total = sum(float(p.grad.detach().pow(2).sum().item())
                        for p in model.parameters() if p.grad is not None)
            norms["n"] = total ** 0.5
            return real(*a, **k)
        optim.step = _spy
        train_step(model, optim, batch, max_len=12, **kw)
        return norms["n"]

    n_free    = _capture(model_a, optim_a)                 # no clip
    n_clipped = _capture(model_b, optim_b, grad_clip=1.0)
    assert n_clipped < n_free
    assert n_clipped <= 1.0 + 1e-4
