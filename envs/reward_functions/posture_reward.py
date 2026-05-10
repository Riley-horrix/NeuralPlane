import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI


class PostureReward(BaseRewardFunction):
    """
    Measure the difference between the current posture and the target posture
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        """
        Args:
            task: task instance
            env: environment instance

        Returns:
            (tensor): reward
        """
        roll, pitch, heading = env.model.get_posture()
        vt = env.model.get_vt()

        # Keep it in radians, don't divide by pi yet
        delta_pitch = wrap_PI(pitch - task.target_pitch)
        delta_heading = wrap_PI(heading - task.target_heading)
        # Normalize velocity error (e.g. 100 ft/s error = 1.0)
        delta_vt = (vt - task.target_vt) / 100.0

        # Use absolute error and multiply by a large scaling factor
        reward_pitch = -torch.abs(delta_pitch)
        reward_heading = -torch.abs(delta_heading)
        reward_vt = -torch.abs(delta_vt)

        # Multiply by 5 so the agent actually feels the penalty
        reward_target = 5.0 * (reward_pitch + reward_heading + reward_vt)

        return reward_target
