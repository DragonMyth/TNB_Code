import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .utils import *

class DartFlatwormSwimStraightEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 36, [-1.0] * 36])
        self.action_scale = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2,np.pi/2] * 6)
        self.torque_scale = np.array([6,3,3,2,2,1]*12)
        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'flatworm.skel', self.frame_skip, 149, control_bounds, dt=0.002,
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
        # self.Kd = 150 * self.simulation_dt * self.Kp
        self.Kd = self.simulation_dt * self.Kp

        self.invM = np.linalg.inv(self.robot_skeleton.M + self.Kd * self.simulation_dt)
        self.symm_rate = -1 * np.array([1, 1, 0.01, 0.01])
        self.constraint_stength = 10
        self.constraint_idx_arr = [1,2,3]

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
        tau[6::]*= self.torque_scale
        tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0

        return self._step_pure_tor(tau)

    def _step_pure_tor(self,tau):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        self.do_simulation(tau, self.frame_skip, self.constraint_stength,self.constraint_idx_arr)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        angs = np.abs(self.robot_skeleton.q[6::])

        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 500

        orth_pen = 0.5 * (np.abs(cur_com[1] - self.original_com[1]) + np.abs(cur_com[2] - self.original_com[2]))

        rotate_pen = np.sum(np.abs(cur_q[:3] - self.original_q[:3]))

        energy_consumed_pen = 0  # 0.1 * np.sum(np.abs(tau[6::] * old_dq[6::] * self.simulation_dt))

        # mirror_enforce
        reward = 1 + horizontal_pos_rwd - rotate_pen - orth_pen - energy_consumed_pen

        notdone = np.isfinite(ob[5::]).all() and (np.abs(angs) < np.pi / 2.0).all()
        done = not notdone

        return ob, reward, done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                  'rotate_pen': -rotate_pen, 'orth_pen': -orth_pen,
                                  'energy_consumed_pen': -energy_consumed_pen, 'tau': tau[6::]}

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

    def do_simulation(self, tau, n_frames, strength, const_arr):

        for _ in range(n_frames):
            comb = []
            import itertools
            for i in itertools.product(['l', 'r'], const_arr):
                comb.append(i)
            for segIdx in range(len(const_arr)*2-1):
                for side, idx in comb:
                    offset1_dir = np.array([-1, 0, 0])
                    offset2_dir = np.array([1, 0, 0])
                    curr_key = 'wing_' + str(side) + '_' + str(segIdx) + str(idx)
                    next_key = 'wing_' + str(side) + '_' + str(segIdx + 1) + str(idx)
                    curr_body = self.bodynodes_dict[curr_key]
                    next_body = self.bodynodes_dict[next_key]

                    constraint_force, offset1, offset2 = calc_constraint_force(curr_body, offset1_dir, next_body,
                                                                                    offset2_dir, strength=strength)

                    curr_body.add_ext_force(constraint_force, _offset=offset1)
                    next_body.add_ext_force(-constraint_force, _offset=offset2)

            super(DartFlatwormSwimStraightEnv,self).do_simulation(tau,1)





    def build_target_pos(self,a):
        target_pos = np.zeros(72)
        a = a * self.action_scale
        target_pos[0:12] = a[0:12]
        target_pos[24:36] = a[12:24]
        target_pos[48:60] = a[24:36]

        return np.concatenate(([0.0] * 6, target_pos))
