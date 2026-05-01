import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class TimeDrivenReward(BaseRewardFunction):
    """
    TimeDrivenReward
    Gives the agent a small positive reward for every step it takes, encouraging it to survive longer.
    """
    def __init__(self, config):
        super().__init__(config)

        self.reward = 0.01

    def get_reward(self, task, env):
        """
        Reward is a small positive value for each step taken.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        return self.reward * ~(env.is_done | env.bad_done)
