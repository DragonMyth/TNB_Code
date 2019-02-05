import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .utils import *
from gym.envs.dart.humanoid_swim_straight import DartHumanoidSwimStraightEnv
class DartHumanoidSwimStraightTauActEnv(DartHumanoidSwimStraightEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 26, [-1.0] * 26])
        self.action_scale = 0.001
        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'humanoid_swimmer.skel', self.frame_skip, 57, control_bounds, dt=0.002,
                                  disableViewer=True,
                                  custom_world=BaseFluidSimulator)
        utils.EzPickle.__init__(self)

        self.init_nec_params()

    def init_nec_params(self):

        self.bodynodes_dict = construct_skel_dict(self.robot_skeleton.bodynodes)

        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

        self.symm_rate = -1 * np.array([1, 1, 0.01, 0.01])
        self.simulation_dt = self.dt * 1.0 / self.frame_skip

        self.constraint_idx_arr = [1,2,3]

    def _step(self, a):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        tau = np.concatenate(([0.0] * 6, a*self.action_scale))


        return self._step_pure_tor(tau)

    def _get_obs(self):

        return np.concatenate([self.robot_skeleton.q[4:8], self.robot_skeleton.dq[3:8], self.robot_skeleton.q[8::],
                               self.robot_skeleton.dq[8::]]).ravel()
