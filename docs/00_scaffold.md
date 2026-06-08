# 00. Scaffold

What this is: the empty shell of the `alpha-rule` package, before any code is
migrated. It gives you an installable package, a license, a test runner, and
CI, so every later component lands on a working foundation.

## Layout

```
alpha-rule/
  pyproject.toml            packaging and optional extras
  LICENSE                   MIT
  README.md                 top level readme
  .gitignore
  .github/workflows/ci.yml  Linux pytest CI
  alpha_rule/               the package (grows component by component)
    __init__.py             version only, for now
    tests/                  test suite + bundled runner
      conftest.py           sys.path bootstrap (fixtures added later)
      run_tests.py          Windows friendly runner
      test_smoke.py         imports the package, checks the version
  docs/                     these docs (a notebook per component from 01 on)
```

## Import name

The project is `alpha-rule`. A hyphen is not valid in a Python import, so the
package you import is `alpha_rule`:

```python
import alpha_rule
print(alpha_rule.__version__)   # "0.1.0"
```

## Install and test

```
pip install -e .[dev]
pytest alpha_rule/tests
# or, on Windows where the pytest console can stall:
python -m alpha_rule.tests.run_tests
```

At this stage the only test is the smoke test (import the package, check the
version). Each later component brings its own tests and its own `docs/` notebook.

## Extras

`pip install -e .` gives you the base (numpy only). Layer on what you need:
`[nn]` for torch, `[rl]` for gymnasium (plus OpenTheChests, separate),
`[analysis]` for plotly/scipy/pandas, `[dev]` for all of it plus the test tools.
