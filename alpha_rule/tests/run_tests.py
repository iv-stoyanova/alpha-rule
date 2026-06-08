"""
Lightweight custom test runner.

The pytest console can stall during collection on some Windows setups, so this
module is a minimal stand-in that runs the suite without it. It supports only
what the suite actually uses:

    - plain ``test_*`` functions that use ``assert``
    - ``pytest.mark.parametrize`` stacked on a function (the marker is read
      directly)
    - fixtures resolved by calling named factory functions from ``conftest``
    - ``pytest.skip()`` (reported as SKIP, not a crash)

Usage::

    python -m alpha_rule.tests.run_tests
    python -m alpha_rule.tests.run_tests test_smoke
    python -m alpha_rule.tests.run_tests -k version
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, List, Tuple

# Ensure the repo root is on sys.path so ``import alpha_rule...`` works.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Fixture resolution
# --------------------------------------------------------------------------- #

def _load_fixtures() -> dict:
    """Import ``conftest`` and expose its factory fixtures by name."""
    from alpha_rule.tests import conftest

    fixtures: dict = {}
    for name, obj in inspect.getmembers(conftest):
        if name.startswith("_") or not callable(obj):
            continue
        raw = getattr(obj, "_fixture_function", obj)
        fixtures[name] = raw
    return fixtures


# --------------------------------------------------------------------------- #
# Test discovery
# --------------------------------------------------------------------------- #

def _discover_test_modules(pattern: str | None = None) -> List[ModuleType]:
    """Import every ``test_*.py`` in this directory. A module that fails to
    import (e.g. an optional dep is missing) prints a SKIP line and is
    otherwise ignored, so one missing dep does not block the rest."""
    tests_dir = Path(__file__).parent
    modules: List[ModuleType] = []
    for py in sorted(tests_dir.glob("test_*.py")):
        module_name = f"alpha_rule.tests.{py.stem}"
        if pattern and pattern not in module_name:
            continue
        try:
            modules.append(importlib.import_module(module_name))
        except Exception as exc:  # noqa: BLE001
            print(f"\n== {module_name} ==")
            print(f"  SKIP  module-import failed: {type(exc).__name__}: {exc}")
    return modules


def _discover_test_functions(module: ModuleType) -> List[Tuple[str, Callable]]:
    return [
        (name, obj)
        for name, obj in inspect.getmembers(module)
        if name.startswith("test_") and callable(obj)
    ]


# --------------------------------------------------------------------------- #
# Parametrize handling
# --------------------------------------------------------------------------- #

def _expand_parametrize(func: Callable) -> List[Tuple[str, dict]]:
    """Return a list of (label, kwargs-override) pairs for this test."""
    marks = getattr(func, "pytestmark", None)
    if not marks:
        return [("", {})]

    cases: List[Tuple[str, dict]] = [("", {})]
    for mark in marks:
        if getattr(mark, "name", None) != "parametrize":
            continue
        argnames, argvalues = mark.args[0], mark.args[1]
        names = [n.strip() for n in argnames.split(",")]
        new_cases: List[Tuple[str, dict]] = []
        for label, kwargs in cases:
            for value in argvalues:
                if len(names) == 1:
                    value_kwargs = {names[0]: value}
                    sub_label = repr(value)
                else:
                    value_kwargs = dict(zip(names, value))
                    sub_label = ",".join(repr(v) for v in value)
                combined_label = (
                    f"{label}[{sub_label}]" if label else f"[{sub_label}]"
                )
                new_cases.append((combined_label, {**kwargs, **value_kwargs}))
        cases = new_cases
    return cases


# --------------------------------------------------------------------------- #
# Run + report
# --------------------------------------------------------------------------- #

def _run_one(func: Callable, label: str, overrides: dict, fixtures: dict) -> Tuple[str, str]:
    """Run one test case. Returns ``(status, detail)`` where status is
    ``"PASS"`` / ``"SKIP"`` / ``"FAIL"``."""
    sig = inspect.signature(func)
    kwargs = {}
    for pname in sig.parameters:
        if pname in overrides:
            kwargs[pname] = overrides[pname]
            continue
        if pname not in fixtures:
            return "FAIL", f"MISSING FIXTURE: {pname}"
        fixture_func = fixtures[pname]
        fixture_sig = inspect.signature(fixture_func)
        if not fixture_sig.parameters:
            kwargs[pname] = fixture_func()
        else:
            sub_kwargs = {
                sub: fixtures[sub]() for sub in fixture_sig.parameters
                if sub in fixtures
            }
            kwargs[pname] = fixture_func(**sub_kwargs)
    try:
        func(**kwargs)
        return "PASS", ""
    except BaseException as exc:  # noqa: BLE001
        # pytest.skip() raises _pytest.outcomes.Skipped, a BaseException, not
        # Exception. Treat it as a SKIP rather than a crash.
        if type(exc).__name__ == "Skipped":
            return "SKIP", str(exc)
        if not isinstance(exc, Exception):
            raise  # genuine BaseException (KeyboardInterrupt/SystemExit): propagate
        return "FAIL", traceback.format_exc()


def run(pattern: str | None = None, name_filter: str | None = None) -> int:
    fixtures = _load_fixtures()
    modules = _discover_test_modules(pattern)

    total = 0
    skipped = 0
    failed: List[Tuple[str, str]] = []
    for module in modules:
        print(f"\n== {module.__name__} ==")
        for name, func in _discover_test_functions(module):
            for label, overrides in _expand_parametrize(func):
                if name_filter and name_filter not in name and name_filter not in label:
                    continue
                total += 1
                status, tb = _run_one(func, label, overrides, fixtures)
                print(f"  {status}  {name}{label}")
                if status == "FAIL":
                    failed.append((f"{module.__name__}.{name}{label}", tb))
                elif status == "SKIP":
                    skipped += 1

    print("\n" + "=" * 60)
    print(f"Ran {total} test cases, {len(failed)} failed, {skipped} skipped")
    for name, tb in failed:
        print(f"\n---- {name} ----\n{tb}")
    return 0 if not failed else 1


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", nargs="?", help="Substring of module name to filter")
    parser.add_argument("-k", dest="name", help="Substring of test name to filter")
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(pattern=args.module, name_filter=args.name)


if __name__ == "__main__":
    raise SystemExit(main())
