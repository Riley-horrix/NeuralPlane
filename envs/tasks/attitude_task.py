import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.dallyverkampen_reward import DallyVerKampenReward
from termination_conditions.unreach_posture import UnreachPosture
from utils.utils import wrap_PI

class AttitudeTask(BaseTask):
    '''
    Inner-loop Attitude Controller Task (14-Dimensional Custom Setup).
    Tracks pitch, roll, sideslip, and manages velocity error for a bundled PID.
    '''
    def __init__(self, config, n, device, random_seed):
        super().__init__(config, n, device, random_seed)

        # Initialize the correct attitude targets (removed heading, added roll and beta)
        self.target_pitch = torch.zeros(self.n, device=self.device)
        self.target_roll = torch.zeros(self.n, device=self.device)
        self.target_beta = torch.zeros(self.n, device=self.device)
        self.target_vt = torch.zeros(self.n, device=self.device)

        # Update increments to match the new targets
        self.max_pitch_increment = getattr(self.config, 'max_pitch_increment', 0.3)
        self.max_roll_increment = getattr(self.config, 'max_roll_increment', 0.3)
        self.max_velocities_u_increment = getattr(self.config, 'max_velocities_u_increment', 100)
        self.noise_scale = getattr(self.config, 'noise_scale', 0.01)

        self.reward_functions = [
            DallyVerKampenReward(self.config)
        ]

        self.termination_conditions = [
            UnreachPosture(self.config, device)
        ]

    def reset(self, env):
        done = env.is_done.bool()
        bad_done = env.bad_done.bool()
        exceed_time_limit = env.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit
        size = torch.sum(reset)

        if size == 0:
            return

        roll, pitch, _ = env.model.get_posture()
        vt = env.model.get_vt()

        # Generate random offsets for the resetting environments
        delta_pitch = 2 * (torch.rand(size, device=self.device) - 0.5) * self.max_pitch_increment
        delta_roll = 2 * (torch.rand(size, device=self.device) - 0.5) * self.max_roll_increment
        delta_vt = 2 * (torch.rand(size, device=self.device) - 0.5) * self.max_velocities_u_increment

        # Assign new targets
        self.target_pitch[reset] = wrap_PI(pitch[reset] + delta_pitch)
        self.target_roll[reset] = wrap_PI(roll[reset] + delta_roll)
        self.target_beta[reset] = 0.0  # Force coordinated flight (zero sideslip)
        self.target_vt[reset] = vt[reset] + delta_vt

    def get_obs(self, env):
        """
        Convert simulation states into the 14-dimensional format of the inner-loop agent.

        observation(dim 14):
            0. weighted_error_beta
            1. weighted_error_pitch
            2. weighted_error_roll
            3. ego_el (elevator deflection)
            4. ego_ail (aileron deflection)
            5. ego_rud (rudder deflection)
            6. ego_P (roll rate)
            7. ego_Q (pitch rate)
            8. ego_R (yaw rate)
            9. vt (velocity)
            10. sin_alpha (angle of attack)
            11. cos_alpha (angle of attack)
            12. EAS2TAS
            13. delta_vt_fts (velocity error in ft/s)
        """
        roll, pitch, _ = env.model.get_posture()
        alpha = env.model.get_AOA()
        beta = env.model.get_AOS()
        P, Q, R = env.model.get_angular_velocity()
        el, ail, rud, _ = env.model.get_control_surface()

        vt = env.model.get_vt()
        norm_vt = vt.reshape(-1, 1) * 0.3048 / 340
        delta_vt_fts = vt - self.target_vt
        eas2tas = env.model.get_EAS2TAS()

        e_beta = wrap_PI(self.target_beta - beta)
        e_pitch = wrap_PI(self.target_pitch - pitch)
        e_roll = wrap_PI(self.target_roll - roll)

        c_beta = (6.0 / torch.pi) * 4.0
        c_pitch = (6.0 / torch.pi) * 1.0
        c_roll = (6.0 / torch.pi) * 1.0

        weighted_e_beta = (e_beta * c_beta).reshape(-1, 1)
        weighted_e_pitch = (e_pitch * c_pitch).reshape(-1, 1)
        weighted_e_roll = (e_roll * c_roll).reshape(-1, 1)

        norm_el = el.reshape(-1, 1) / 45.0
        norm_ail = ail.reshape(-1, 1) / 45.0
        norm_rud = rud.reshape(-1, 1) / 45.0

        norm_P = P.reshape(-1, 1)
        norm_Q = Q.reshape(-1, 1)
        norm_R = R.reshape(-1, 1)

        alpha_sin = torch.sin(alpha.reshape(-1, 1))
        alpha_cos = torch.cos(alpha.reshape(-1, 1))

        obs = torch.hstack((
            weighted_e_beta, weighted_e_pitch, weighted_e_roll,
            norm_el, norm_ail, norm_rud,
            norm_P, norm_Q, norm_R,
            norm_vt,
            alpha_sin, alpha_cos,
            eas2tas.reshape(-1, 1),
            delta_vt_fts.reshape(-1, 1)
        ))

        return obs + torch.randn_like(obs) * self.noise_scale