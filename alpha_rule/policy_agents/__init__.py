"""
Policy agents that ``RuleSimulator`` trains to score a candidate rule.

The reinforcement-learning reward backend lives here: a candidate rule is
turned into an observation wrapper, an agent is trained in the wrapped
environment, and an evaluation function reads back a scalar score. The only
backend so far is tabular Q-learning, in the ``q_learning`` subpackage.

Part of the optional ``[rl]`` extra (needs ``gymnasium`` plus a runtime
environment such as OpenTheChests).
"""
