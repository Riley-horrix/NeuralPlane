import torch
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from reward_function_base import BaseRewardFunction
from utils.utils import wrap_PI

class DallyVerKampenReward(BaseRewardFunction):
    """
    Measures the difference between the current posture and the target posture,
    incorporating clipped bounding, aerodynamic safety penalties, and a survival bonus.
    """
    def __init__(self, config):
        super().__init__(config)

        # 1. Cost weights for tracking errors
        self.cost_pitch = getattr(self.config, 'cost_pitch', 2.0)
        self.cost_heading = getattr(self.config, 'cost_heading', 2.0)
        self.cost_vt = getattr(self.config, 'cost_vt', 1.0)

        # 2. Angle of Attack (AoA) limits for F-16 (converted to radians)
        # Warning starts at 15 deg, hard stall limit is ~25 deg
        self.aoa_warning_rad = math.radians(15.0)
        self.aoa_max_rad = math.radians(25.0)

    def get_reward(self, task, env):
        """
        Args:
            task: task instance
            env: environment instance
        Returns:
            (tensor): reward
        """
        # Extract current states
        roll, pitch, heading = env.model.get_posture()
        vt = env.model.get_vt()
        aoa = env.model.get_AOA()

        # Calculate raw errors
        delta_pitch = wrap_PI(pitch - task.target_pitch)
        delta_heading = wrap_PI(heading - task.target_heading)
        # Normalize velocity error (e.g., 100 ft/s error = 1.0)
        delta_vt = (vt - task.target_vt) / 100.0

        # Apply cost weights and absolute value
        weighted_pitch_err = self.cost_pitch * torch.abs(delta_pitch)
        weighted_heading_err = self.cost_heading * torch.abs(delta_heading)
        weighted_vt_err = self.cost_vt * torch.abs(delta_vt)

        # Clip the errors. We negate them and clamp between -1.0 and 0.0
        # This bounds the maximum penalty per step, stopping exploding Q-values
        clip_pitch = torch.clamp(-weighted_pitch_err, min=-1.0, max=0.0)
        clip_heading = torch.clamp(-weighted_heading_err, min=-1.0, max=0.0)
        clip_vt = torch.clamp(-weighted_vt_err, min=-1.0, max=0.0)

        # Tracking reward is now strictly bounded between -3.0 and 0.0
        tracking_reward = clip_pitch + clip_heading + clip_vt

        # Exponentially penalize the agent if it pulls too much pitch and approaches a stall
        aoa_penalty = torch.zeros_like(aoa)

        # Take absolute value of AoA because negative AoA (pushing the nose down) also has limits
        abs_aoa = torch.abs(aoa)
        violation_mask = abs_aoa > self.aoa_warning_rad

        # Quadratic penalty that scales up sharply as it reaches the stall limit
        # At 15 deg, penalty is 0.0. At 25 deg, penalty is -5.0.
        if violation_mask.any():
            aoa_penalty[violation_mask] = -5.0 * ((abs_aoa[violation_mask] - self.aoa_warning_rad) / (self.aoa_max_rad - self.aoa_warning_rad))**2

        # Because tracking_reward is [-3.0, 0.0], we add a flat +3.0.
        # This shifts the base tracking step reward to [0.0, +3.0].
        # By making safe flight mathematically positive, we eliminate the "suicide problem".
        survival_bonus = 3.0

        # Final summation
        total_reward = survival_bonus + tracking_reward + aoa_penalty

        return total_reward