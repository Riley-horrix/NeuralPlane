import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Done: +1
    - Bad_done: -1
    - Exceed_time_limit: +1
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        reward = -5 * env.bad_done + 5 * env.is_done + 5 * env.exceed_time_limit
        return reward
