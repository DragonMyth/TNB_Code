import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .utils import calc_constraint_force
from .flatworm_swim_straight import  DartFlatwormSwimStraightEnv

class DartFlatwormSwimStraightSmallEnv(DartFlatwormSwimStraightEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 16, [-1.0] * 16])
        self.action_scale = np.array([2 * np.pi, 2*np.pi, np.pi, np.pi] * 4)
        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'flatworm_sm.skel', self.frame_skip, 69, control_bounds, dt=0.002,
                                  disableViewer=not True,
                                  custom_world=BaseFluidSimulator)
        utils.EzPickle.__init__(self)

        self.init_nec_params()
        self.constraint_stength = 10
        self.constraint_idx_arr = [1,2]

    def _step(self, a):


        target_pos = self.build_target_pos(a)
        invM = self.invM
        p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
        d = -self.Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.simulation_dt

        tau *= 0.001
        tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0

        self._step_pure_tor(tau)

    def _get_obs(self):

        return np.concatenate([self.robot_skeleton.q[4:6], self.robot_skeleton.dq[3:6], self.robot_skeleton.q[6::],
                               self.robot_skeleton.dq[6::]]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(-60)
        self.track_skeleton_id = 0

    def build_target_pos(self, a):
        target_pos = np.zeros(32)
        a = a * self.action_scale
        target_pos[0:4] = a[0:4]
        target_pos[4:8] = a[4:8]

        target_pos[16:20] = a[8:12]
        target_pos[20:24] = a[12:16]

        return np.concatenate(([0.0] * 6, target_pos))

    def build_torque(self, a):
        target_torque = np.zeros(32)
        target_torque[0:4] = a[0:4] * self.action_scale
        target_torque[4:8] = a[4:8] * self.action_scale
        target_torque[16:20] = a[8:12] * self.action_scale
        target_torque[20:24] = a[12:16] * self.action_scale
        return np.concatenate(([0.0] * 6, target_torque))


class DartFlatwormSwimStraightSmallFreeBackEnv(DartFlatwormSwimStraightEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 19, [-1.0] * 19])
        self.action_scale = np.pi / 2
        # self.action_scale = np.concatenate((np.array([np.pi,np.pi,np.pi]) ,
        #                                    np.array([2 * np.pi, 2*np.pi, np.pi, np.pi * 0.5] * 4)))
        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'flatworm_sm_freeback.skel', self.frame_skip, 75, control_bounds, dt=0.002,
                                  disableViewer=not True,
                                  custom_world=BaseFluidSimulator)
        utils.EzPickle.__init__(self)


        self.init_nec_params()

        self.constraint_stength = 10
        self.constraint_idx_arr = [1,2]

    def _step(self, a):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        target_pos = self.build_target_pos(a)
        invM = self.invM
        p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
        d = -self.Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.simulation_dt

        tau *= 0.001
        tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0

        return self._step_pure_tor(tau)
    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q[4:9], self.robot_skeleton.dq[3:9], self.robot_skeleton.q[9::],
                               self.robot_skeleton.dq[9::]]).ravel()

    def build_torque(self, a):
        target_torque = np.zeros(35)
        a = a * self.action_scale
        target_torque[0:3] = a[0:3]
        target_torque[3:7] = a[3:7]
        target_torque[7:11] = a[7:11]
        target_torque[19:23] = a[11:15]
        target_torque[23:27] = a[15:19]
        return np.concatenate(([0.0] * 6, target_torque))

    def build_target_pos(self, a):
        target_pos = np.zeros(35)
        a = a * self.action_scale

        target_pos[0:3] = a[0:3]
        target_pos[0+3:4+3] = a[0+3:4+3]
        target_pos[4+3:8+3] = a[4+3:8+3]

        target_pos[16+3:20+3] = a[8+3:12+3]
        target_pos[20+3:24+3] = a[12+3:16+3]

        return np.concatenate(([0.0] * 6, target_pos))


class DartFlatwormSwimStraightNoSymmEnv(DartFlatwormSwimStraightSmallEnv, utils.EzPickle):
    def _step(self, a):
        self.symm_rate = 0
        super(DartFlatwormSwimStraightNoSymmEnv, self)._step(self, a)
