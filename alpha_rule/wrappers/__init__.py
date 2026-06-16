"""
Gym wrappers for the rule-search RL backend.

``HistoryToRuleWrapperBase`` (in ``history_to_rule``) keeps a rolling window of
recent events and replaces the environment observation with a binary vector
saying which of the supplied Allen rules currently match that window. It is the
bridge between the symbolic rule core and the reinforcement-learning
environment.

``OneHotBoxActionWrapper`` (in ``box_action``) restricts the OpenTheChests
button-mask action space to "open one box or none", shrinking it from
``2**n_boxes`` to ``n_boxes + 1``.

Import the wrappers from their modules so this package stays import-light::

    from alpha_rule.wrappers.history_to_rule import HistoryToRuleWrapperBase
    from alpha_rule.wrappers.box_action import OneHotBoxActionWrapper

Part of the optional ``[rl]`` extra (needs ``gymnasium``).
"""
