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

        # F-16 Asymmetric AoA Limits (in radians)
        pos_warning = math.radians(15.0)
        pos_max = math.radians(25.0)

        neg_warning = math.radians(-5.0)
        neg_max = math.radians(-10.0)

        aoa_penalty = torch.zeros_like(aoa)

        # 1. Positive AoA Penalty (Pulling up)
        pos_mask = aoa > pos_warning
        if pos_mask.any():
            aoa_penalty[pos_mask] = -5.0 * ((aoa[pos_mask] - pos_warning) / (pos_max - pos_warning))**2

        # 2. Negative AoA Penalty (Pushing down)
        neg_mask = aoa < neg_warning
        if neg_mask.any():
            # Note: order of subtraction flipped to keep the fraction positive before squaring
            aoa_penalty[neg_mask] = -5.0 * ((aoa[neg_mask] - neg_warning) / (neg_max - neg_warning))**2

        # Because tracking_reward is [-1.0, 0.0], we add a flat +1.0.
        # This shifts the base tracking step reward to [0.0, +1.0].
        # By making safe flight mathematically positive, we eliminate the "suicide problem".
        survival_bonus = 1.0

        # Final summation
        total_reward = survival_bonus + tracking_reward + aoa_penalty

        return total_reward