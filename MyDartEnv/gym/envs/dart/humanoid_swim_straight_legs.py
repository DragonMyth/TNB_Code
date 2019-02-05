import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .utils import *


class DartHumanoidSwimStraightLegsEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 12, [-1.0] * 12])
        self.action_scale = np.pi / 2
        self.torque_scale = 1
        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'humanoid_swimmer_legs.skel', self.frame_skip, 29, control_bounds, dt=0.002,
                                  disableViewer=True,
                                  custom_world=BaseFluidSimulator)
        utils.EzPickle.__init__(self)

        self.init_nec_params()

    def init_nec_params(self):
        self.bodynodes_dict = construct_skel_dict(self.robot_skeleton.bodynodes)

        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

        num_of_dofs = len(self.robot_skeleton.dofs) - len(self.robot_skeleton.joints[0].dofs)
        #
        self.simulation_dt = self.dt * 1.0 / self.frame_skip
        self.Kp = np.diagflat([0.0] * len(self.robot_skeleton.joints[0].dofs) + [4000.0] * num_of_dofs)
        self.Kd = self.simulation_dt * self.Kp

        self.invM = np.linalg.inv(self.robot_skeleton.M + self.Kd * self.simulation_dt)
        self.symm_rate = 2 * np.array([1, 1, 1, 0.8,0.3,0.3])

        self.constraint_idx_arr = [1, 2, 3]

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

        tau *= 0.0003
        tau[6::] *= self.torque_scale
        tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0

        return self._step_pure_tor(tau)

    def _step_pure_tor(self, tau):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        self.do_simulation(tau, self.frame_skip)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        angs = np.abs(self.robot_skeleton.q[6::])

        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 5000

        orth_pen = 0.5 * (np.abs(cur_com[1] - self.original_com[1]) + np.abs(cur_com[2] - self.original_com[2]))

        rotate_pen = np.sum(np.abs(cur_q[:3] - self.original_q[:3]))

        energy_consumed_pen = 100 * np.sum(np.abs(tau[6::] * old_dq[6::] * self.simulation_dt))

        symm_pos_pen = np.sum(self.symm_rate *
                              np.abs(cur_q['left_femur_root_joint_z',
                                           'left_femur_root_joint_y',
                                           'left_femur_root_joint_x',
                                           'left_fibula_femur_joint',
                                           'left_tarsals_febula_joint_1',
                                           'left_tarsals_febula_joint_2'] -
                                     cur_q['right_femur_root_joint_z',
                                           'right_femur_root_joint_y',
                                           'right_femur_root_joint_x',
                                           'right_fibula_femur_joint',
                                           'right_tarsals_febula_joint_1',
                                           'right_tarsals_febula_joint_2'
                                     ]))

        # mirror_enforce
        reward = 1 + horizontal_pos_rwd - rotate_pen - orth_pen - energy_consumed_pen - symm_pos_pen

        notdone = np.isfinite(ob[5::]).all() and (np.abs(angs) < np.pi).all()
        done = not notdone

        return ob, reward, done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                  'rotate_pen': -rotate_pen, 'orth_pen': -orth_pen,
                                  'energy_consumed_pen': -energy_consumed_pen, 'tau': tau[6::],'symm_pos_pens': -symm_pos_pen}

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
        self._get_viewer().scene.tb._set_theta(60)
        self.track_skeleton_id = 0

    def build_target_pos(self, a):
        target_pos = np.zeros(12)
        a = a * self.action_scale
        target_pos = a[:]

        return np.concatenate(([0.0] * 6, target_pos))
