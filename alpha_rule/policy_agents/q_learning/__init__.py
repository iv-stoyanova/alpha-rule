"""
Tabular Q-learning backend: the RL agent ``RuleSimulator`` trains to
score a candidate rule.

Split by concern:

    - ``builder``      : agent construction / training
      (``q_learning_agent_builder``, ``get_state_tuple``)
    - ``eval_total_reward`` / ``eval_mean_success`` /
      ``eval_recognition_distance`` / ``eval_ground_truth_distance`` :
      independent agent-evaluation functions
    - ``play``         : single-episode play helper for debugging

Import from ``alpha_rule.policy_agents.q_learning`` (this package
re-exports each public symbol below).
"""
from alpha_rule.policy_agents.q_learning.builder import (  # noqa: F401
    get_state_tuple,
    q_learning_agent_builder,
)
from alpha_rule.policy_agents.q_learning.eval_total_reward import (  # noqa: F401
    q_learning_agent_eval_total_reward,
)
from alpha_rule.policy_agents.q_learning.eval_mean_success import (  # noqa: F401
    q_learning_agent_eval_mean_reward_success_steps,
)
from alpha_rule.policy_agents.q_learning.eval_recognition_distance import (  # noqa: F401
    q_learning_agent_eval_recognition_distance,
)
from alpha_rule.policy_agents.q_learning.eval_ground_truth_distance import (  # noqa: F401
    q_learning_agent_eval_ground_truth_distance,
)
from alpha_rule.policy_agents.q_learning.play import (  # noqa: F401
    sample_agent_play_one_episode,
)
