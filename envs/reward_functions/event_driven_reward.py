import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class EventDrivenReward(BaseRewardFunction):
    """
    Provides massive, unambiguous terminal signals so the Critic clearly understands
    the ultimate goal (Success) vs. the ultimate failure (Crash/Stall).
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        # Must cast to float so PyTorch can do the math
        is_done = env.is_done.float()
        bad_done = env.bad_done.float()

        # Massive terminal spikes
        reward = (-100.0 * bad_done) + (100.0 * is_done)

        return reward