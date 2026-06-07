import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class PositionReward(BaseRewardFunction):
    """
    Rewards the agent for proximity to the target using a bounded exponential curve.
    At 0km distance, reward is +1.0. As distance increases, reward smoothly approaches 0.0.
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        npos, epos, altitude = env.model.get_position()

        # Calculate deltas in raw units, convert to kilometers
        delta_n = (npos - task.target_npos) * 0.3048 / 1000.0
        delta_e = (epos - task.target_epos) * 0.3048 / 1000.0
        delta_alt = (altitude - task.target_altitude) * 0.3048 / 1000.0

        # True 3D Euclidean distance in km
        dist_km = torch.sqrt(delta_n**2 + delta_e**2 + delta_alt**2)

        # Exponential decay: hyperparameter '2.0' dictates how wide the reward radius is.
        # A higher denominator makes the reward drop off slower, guiding the agent from further away.
        reward_target = torch.exp(-dist_km / 2.0)

        return reward_target