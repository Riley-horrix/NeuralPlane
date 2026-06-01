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
    Strictly follows the attitude tracking math (beta, pitch, roll) from Dally & van Kampen (2022).
    """
    def __init__(self, config):
        super().__init__(config)

        self.max_vt_error = getattr(self.config, 'max_velocities_u_increment', 100.0)

        # 1. Cost weights for tracking errors based on the paper's c^{att} vector: 6/pi * [4, 1, 1]
        self.cost_beta = getattr(self.config, 'cost_beta', (6.0 / math.pi) * 4.0)
        self.cost_pitch = getattr(self.config, 'cost_pitch', (6.0 / math.pi) * 1.0)
        self.cost_roll = getattr(self.config, 'cost_roll', (6.0 / math.pi) * 1.0)
        base_vel_weight = (6.0 / math.pi) * 0.5
        self.cost_vel = getattr(self.config, 'cost_vel', base_vel_weight / self.max_vt_error)

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
        roll, pitch, _ = env.model.get_posture()
        beta = env.model.get_AOS()
        aoa = env.model.get_AOA()
        vt = env.model.get_vt()

        # Calculate raw errors (Target - Actual)
        delta_beta = wrap_PI(task.target_beta - beta)
        delta_pitch = wrap_PI(task.target_pitch - pitch)
        delta_roll = wrap_PI(task.target_roll - roll)
        delta_vt_fts = vt - task.target_vt

        # Apply cost weights and absolute value
        weighted_beta_err = self.cost_beta * torch.abs(delta_beta)
        weighted_pitch_err = self.cost_pitch * torch.abs(delta_pitch)
        weighted_roll_err = self.cost_roll * torch.abs(delta_roll)
        weighted_vt_err = self.cost_vel * torch.abs(delta_vt_fts)

        # Clip the errors. We negate them and clamp between -1.0 and 0.0
        # This bounds the maximum penalty per step, stopping exploding Q-values
        clip_beta = torch.clamp(-weighted_beta_err, min=-1.0, max=0.0)
        clip_pitch = torch.clamp(-weighted_pitch_err, min=-1.0, max=0.0)
        clip_roll = torch.clamp(-weighted_roll_err, min=-1.0, max=0.0)
        clip_vt = torch.clamp(-weighted_vt_err, min=-1.0, max=0.0)

        # The paper divides the L1 norm by 4, so the tracking reward is bounded between -1.0 and 0.0
        tracking_reward = (clip_beta + clip_pitch + clip_roll + clip_vt) / 4.0

        # Exponentially penalize the agent if it pulls too much pitch and approaches a stall
        aoa_penalty = torch.zeros_like(aoa)

        # Take absolute value of AoA because negative AoA (pushing the nose down) also has limits
        abs_aoa = torch.abs(aoa)
        violation_mask = abs_aoa > self.aoa_warning_rad

        # Quadratic penalty that scales up sharply as it reaches the stall limit
        # At 15 deg, penalty is 0.0. At 25 deg, penalty is -5.0.
        if violation_mask.any():
            aoa_penalty[violation_mask] = -5.0 * ((abs_aoa[violation_mask] - self.aoa_warning_rad) / (self.aoa_max_rad - self.aoa_warning_rad))**2

        # Because tracking_reward is [-1.0, 0.0], we add a flat +1.0.
        # This shifts the base tracking step reward to [0.0, +1.0].
        # By making safe flight mathematically positive, we eliminate the "suicide problem".
        survival_bonus = 1.0

        # Final summation
        total_reward = survival_bonus + tracking_reward + aoa_penalty

        return total_reward