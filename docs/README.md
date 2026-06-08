# Component docs

One notebook per migrated component. Each one explains the component, shows how
to use it, and runs a few basic checks you can execute top to bottom.

| Doc | Component | Needs |
|-----|-----------|-------|
| 00_scaffold.md | Project layout, install, tests | base |
| 01_symbolic_core.ipynb | Allen matrix and rule matching | base |
| 02_grammar.ipynb | Grammar and core contracts | base |
| 03_mcts.ipynb | MCTS search engine | base |
| 04_nn.ipynb | Policy and value network | nn |
| 05_neural_evaluator.ipynb | Neural evaluator | nn |
| 06_training.ipynb | Training loop and logging | nn |
| 07_rl_backend.ipynb | Reinforcement learning reward backend | rl + OpenTheChests |
| 08_analysis.ipynb | Plots and statistics | analysis |

Entries past 00 arrive as their components are migrated.
