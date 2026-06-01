import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from task_base import BaseTask
from reward_functions.dallyverkampen_reward import DallyVerKampenReward
from reward_functions.event_driven_reward import EventBasedReward, EventDrivenReward
from termination_conditions.unreach_posture import UnreachPosture
from termination_conditions.overload import Overload
from termination_conditions.high_speed import HighSpeed
from termination_conditions.low_speed import LowSpeed
from termination_conditions.extreme_state import ExtremeState
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
            DallyVerKampenReward(self.config),
            EventDrivenReward(self.config)
        ]

        self.termination_conditions = [
            Overload(self.config),
            HighSpeed(self.config),
            LowSpeed(self.config),
            ExtremeState(self.config),
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
        Convert simulation states into a clean 11-dimensional format for 4-DoF control.

        observation(dim 11):
            0. weighted_error_beta
            1. weighted_error_pitch
            2. weighted_error_roll
            3. weighted_error_vel
            4. norm_el  (elevator deflection)
            5. norm_ail (aileron deflection)
            6. norm_rud (rudder deflection)
            7. norm_thr (throttle setting)
            8. ego_P    (roll rate)
            9. ego_Q    (pitch rate)
            10. ego_R   (yaw rate)
        """
        roll, pitch, _ = env.model.get_posture()
        beta = env.model.get_AOS()
        P, Q, R = env.model.get_angular_velocity()

        # Unpack all 4 control surface signals from the underlying F-16 flight engine
        el, ail, rud, thr = env.model.get_control_surface()
        vt = env.model.get_vt()

        # Calculate tracking states
        e_beta = wrap_PI(self.target_beta - beta)
        e_pitch = wrap_PI(self.target_pitch - pitch)
        e_roll = wrap_PI(self.target_roll - roll)
        delta_vt_fts = vt - self.target_vt

        # Compute weighted states
        weighted_e_beta = (e_beta * (6.0 / torch.pi) * 4.0).reshape(-1, 1)
        weighted_e_pitch = (e_pitch * (6.0 / torch.pi) * 1.0).reshape(-1, 1)
        weighted_e_roll = (e_roll * (6.0 / torch.pi) * 1.0).reshape(-1, 1)

        # Use your explicit max error bounds parameter to scale the state observation cleanly
        weighted_error_vel = (delta_vt_fts * ((6.0 / torch.pi) * 0.5) / self.max_velocities_u_increment).reshape(-1, 1)

        # Normalize physical actuation ranges to a consistent scale of [-1.0, 1.0]
        norm_el = el.reshape(-1, 1) / 45.0
        norm_ail = ail.reshape(-1, 1) / 45.0
        norm_rud = rud.reshape(-1, 1) / 45.0
        norm_thr = (thr.reshape(-1, 1) * 2.0) - 1.0  # Maps raw [0.0, 1.0] throttle to [-1.0, 1.0]

        # Kinematic dampening tracking states
        norm_P = P.reshape(-1, 1)
        norm_Q = Q.reshape(-1, 1)
        norm_R = R.reshape(-1, 1)

        # Stacks into a unified 11-dimensional tensor
        obs = torch.hstack((
            weighted_e_beta, weighted_e_pitch, weighted_e_roll, weighted_error_vel,
            norm_el, norm_ail, norm_rud, norm_thr,
            norm_P, norm_Q, norm_R
        ))

        return obs + torch.randn_like(obs) * self.noise_scale