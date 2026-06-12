"""
Gym observation wrappers that turn a candidate rule into a reward signal.

``HistoryToRuleWrapperBase`` (in ``history_to_rule``) keeps a rolling window of
recent events and replaces the environment observation with a binary vector
saying which of the supplied Allen rules currently match that window. It is the
bridge between the symbolic rule core and the reinforcement-learning
environment.

Import the wrapper from its module so this package stays import-light::

    from alpha_rule.wrappers.history_to_rule import HistoryToRuleWrapperBase

Part of the optional ``[rl]`` extra (needs ``gymnasium``).
"""
