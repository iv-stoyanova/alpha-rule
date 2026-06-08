"""Pytest configuration for the alpha-rule test suite.

Adds the repo root to ``sys.path`` so ``import alpha_rule`` resolves whether
the suite is run via pytest or the bundled runner. Shared fixtures are added
here as the components that need them are migrated: the event / history
fixtures arrive with the symbolic core, the fake gym env with the reinforcement
learning backend.
"""
from __future__ import annotations

import sys
from pathlib import Path

# tests/ -> alpha_rule/ -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
