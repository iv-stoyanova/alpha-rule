# alpha-rule

AlphaZero style Monte Carlo Tree Search over Allen interval temporal rules. A
policy and value network guides the search across a grammar of rules, and each
candidate rule is scored by training a small reinforcement learning agent in a
rule wrapped environment.

The repo is built up one component at a time. `docs/` holds a notebook per
component (what it is, how to use it, basic checks). `alpha_rule/README.md`
holds the package level reference once the code lands.

## Import name

The project is `alpha-rule`. Python import names cannot contain a hyphen, so
the package you import is `alpha_rule`:

```python
import alpha_rule
print(alpha_rule.__version__)
```

## Install

The core (grammar, MCTS, symbolic rule matching) needs only numpy:

```
pip install -e .
```

Optional extras layer on the heavier pieces:

```
pip install -e .[nn]        # policy/value network and training loop (torch)
pip install -e .[rl]        # reinforcement learning reward backend (gymnasium)
pip install -e .[analysis]  # plots and statistics (plotly, scipy, pandas)
pip install -e .[dev]       # all of the above plus the test tooling
```

The `rl` backend also needs OpenTheChests, installed separately (see
`docs/07_rl_backend.ipynb`). The package never imports it; it only expects a
gym environment that exposes `get_otc()` and `get_types()`.

## Tests

```
pytest alpha_rule/tests
```

On Windows, where the pytest console can stall, use the bundled runner instead:

```
python -m alpha_rule.tests.run_tests
python -m alpha_rule.tests.run_tests -k smoke
```

## License

MIT. See `LICENSE`.
