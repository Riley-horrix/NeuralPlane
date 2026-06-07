import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from reward_function_base import BaseRewardFunction


class InnerEventDrivenReward(BaseRewardFunction):
    """
    Event reward strictly for continuous inner-loop tracking.
    Penalizes catastrophic aerodynamic departures (bad_done).
    Ignores standard timeouts (is_done) because there is no 'finish line'.
    """
    def __init__(self, config):
        super().__init__(config)

    def get_reward(self, task, env):
        # Cast to float for tensor math
        bad_done = env.bad_done.float()

        # A moderate terminal penalty for crashing.
        # -10.0 is large enough to deter crashes, but small enough
        # not to drown out the continuous DallyVerKampen tracking gradients.
        reward = -10.0 * bad_done

        return reward