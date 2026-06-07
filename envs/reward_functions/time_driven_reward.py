import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class TimeDrivenReward(BaseRewardFunction):
    """
    Applies a small, constant time penalty to prevent 'loitering' or circling.
    The agent is incentivized to minimize flight time to the target.
    """
    def __init__(self, config):
        super().__init__(config)
        # Small negative penalty per step
        self.step_penalty = -0.05

    def get_reward(self, task, env):
        is_done = env.is_done.bool()
        bad_done = env.bad_done.bool()

        # Apply penalty only if the episode is still actively running
        active_mask = ~(is_done | bad_done)

        return self.step_penalty * active_mask.float()