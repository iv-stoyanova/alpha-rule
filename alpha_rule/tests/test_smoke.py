"""Scaffold smoke test: the package imports and exposes a version string.

This is the only test at the scaffold stage. Real per-component tests arrive
with their components.
"""
from __future__ import annotations


def test_package_imports():
    import alpha_rule
    assert alpha_rule is not None


def test_version_is_a_nonempty_string():
    import alpha_rule
    assert isinstance(alpha_rule.__version__, str)
    assert alpha_rule.__version__
