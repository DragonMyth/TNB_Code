import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .simple_water_world import BaseFluidEnhancedSimulator
from keras.models import load_model


class DartTurtleSwimStraighSPDEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 8, [-1.0] * 8])
        self.action_scale = np.pi / 2.0
        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'large_flipper_turtle_real.skel', self.frame_skip, 21, control_bounds, dt=0.002,
                                  disableViewer=not True,
                                  custom_world=BaseFluidEnhancedSimulator)

        utils.EzPickle.__init__(self)
        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

        num_of_dofs = len(self.robot_skeleton.dofs) - len(self.robot_skeleton.joints[0].dofs)

        self.simulation_dt = self.dt * 1.0 / self.frame_skip
        self.Kp = np.diagflat([0.0] * len(self.robot_skeleton.joints[0].dofs) + [4000.0] * num_of_dofs)
        # self.Kd = 150 * self.simulation_dt * self.Kp
        self.Kd = 20 * self.simulation_dt * self.Kp
        self.invM = np.linalg.inv(self.robot_skeleton.M + self.Kd * self.simulation_dt)

        # Symm rate Positive to be enforcing Symmetry
        #          Negative to be enforcing Asymmetry
        self.symm_rate = -0 * np.array([1, 1, 0.01, 0.01])

        self.stepNum = 0
        self.recordGap = 3
        self.traj_buffer = []
        self.symm_autoencoder = load_model("./AutoEncoder/autoencoder_one_step.h5")
        self.novelty_factor = 10
        print("Model name is", self.symm_autoencoder.name)

    def _step(self, a):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        target_pos = np.concatenate(([0.0] * 6, a * self.action_scale))
        ##SPD Controller
        # for i in range(self.frame_skip):
        #     invM = self.invM
        #     p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
        #     d = -self.Kd.dot(self.robot_skeleton.dq)
        #     qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        #     tau = p + d - self.Kd.dot(qddot) * self.simulation_dt
        #
        #     tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
        #     self.do_simulation(tau, 1)

        invM = self.invM
        p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
        d = -self.Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.simulation_dt

        tau /= 100.0

        tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
        self.do_simulation(tau, self.frame_skip)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        # this should be the unscaled novelty term
        novelDiff = 0
        if (self.stepNum % self.recordGap == 0):
            # 5 here is the num of dim for root related states
            self.traj_buffer.append(ob[5::])
        if (len(self.traj_buffer) == 15):
            # Reshape to 1 dimension
            traj_seg = np.array([self.traj_buffer])
            traj_seg = self.normalizeTraj(traj_seg, -np.pi / 2, np.pi / 2, -50, 50)
            traj_seg = traj_seg.reshape((len(traj_seg), np.prod(traj_seg.shape[1:])))

            traj_recons = self.symm_autoencoder.predict(traj_seg)
            diff = traj_recons - traj_seg
            novelDiff = np.linalg.norm(diff[:], axis=1)[0]
            # print("Novel diff is", nov elDiff)
            self.traj_buffer.pop(0)

        self.stepNum += 1

        novelDiff = self.novelty_factor * novelDiff * 2
        angs = np.abs(self.robot_skeleton.q[6::])
        ang_cost_rate = 0.0001
        torque_cost_rate = 0.001
        symm_pos_pen = np.sum(self.symm_rate *
                              np.abs(cur_q['left_front_limb_joint_1',
                                           'left_front_limb_joint_2',
                                           'left_rear_limb_joint_1',
                                           'left_rear_limb_joint_2'] -
                                     cur_q['right_front_limb_joint_1',
                                           'right_front_limb_joint_2',
                                           'right_rear_limb_joint_1',
                                           'right_rear_limb_joint_2']))

        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 500
        horizontal_vel_rwd = 0  # 3*cur_dq[3]
        orth_pen = 0.5 * (np.abs(cur_com[1] - self.original_com[1]) + np.abs(cur_com[2] - self.original_com[2]))
        rotate_pen = 0.1 * np.sum(np.abs(cur_q[0]))
        # mirror_enforce
        reward = 1 + horizontal_pos_rwd + horizontal_vel_rwd - rotate_pen - orth_pen - symm_pos_pen + novelDiff

        valid = np.isfinite(ob[5::]).all() and (np.abs(angs) < np.pi / 2.0).all() and (np.abs(cur_dq) < 50).all()
        done = not valid

        return ob, reward, done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                  'horizontal_vel_rwd': horizontal_vel_rwd,
                                  'rotate_pen': -rotate_pen, 'orth_pen': -orth_pen, 'tau': tau,
                                  'symm_pos_pens': -symm_pos_pen, 'novelty': novelDiff}

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

    def normalizeTraj(self, traj, minq, maxq, mindq, maxdq):
        traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (maxq - minq)
        traj[:, :, int(len(traj[0, 0]) / 2)::] /= (maxdq - mindq)
        return traj


class DartTurtleSwimStraighSPDEnvNoEnf(DartTurtleSwimStraighSPDEnv):
    def __init__(self):
        DartTurtleSwimStraighSPDEnv.__init__(self)
        self.symm_rate = -0 * np.array([1, 1, 0.01, 0.01])

    def _step(self, a):
        self.symm_rate = -5 * np.array([1, 1, 0.01, 0.01])

        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq
        target_pos = np.concatenate(([0.0] * 6, a * self.action_scale))

        # for i in range(self.frame_skip):
        #     invM = self.invM
        #     p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
        #     d = -self.Kd.dot(self.robot_skeleton.dq)
        #     qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        #     tau = p + d - self.Kd.dot(qddot) * self.simulation_dt
        #     # tau *= 0.0005
        #     tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
        #     self.do_simulation(tau, 1)

        invM = self.invM
        p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
        d = -self.Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.simulation_dt
        tau /= 100.0
        tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
        self.do_simulation(tau, self.frame_skip)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        # # this should be the unscaled novelty term
        novelDiff = 0
        # if (self.stepNum % self.recordGap == 0):
        #     # 5 here is the num of dim for root related states
        #     self.traj_buffer.append(ob[5::])
        # if (len(self.traj_buffer) == 15):
        #     # Reshape to 1 dimension
        #     traj_seg = np.array([self.traj_buffer])
        #     traj_seg = self.normalizeTraj(traj_seg, -np.pi / 2, np.pi / 2, -50, 50)
        #     traj_seg = traj_seg.reshape((len(traj_seg), np.prod(traj_seg.shape[1:])))
        #
        #     traj_recons = self.symm_autoencoder.predict(traj_seg)
        #     diff = traj_recons - traj_seg
        #     novelDiff = np.linalg.norm(diff[:], axis=1)[0]
        #     # print("Novel diff is", nov elDiff)
        #     self.traj_buffer.pop(0)
        #
        # self.stepNum += 1
        # novelDiff = self.novelty_factor * novelDiff

        angs = np.abs(self.robot_skeleton.q[6::])
        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 1000
        horizontal_vel_rwd = 0  # 3*cur_dq[3]
        orth_pen = 0.5 * (np.abs(cur_com[1] - self.original_com[1]) + np.abs(cur_com[2] - self.original_com[2]))
        rotate_pen = 0.1 * np.sum(np.abs(cur_q[0]))

        symm_pos_pen = 0.1 * np.sum(self.symm_rate *
                                    np.abs(cur_q['left_front_limb_joint_1',
                                                 'left_front_limb_joint_2',
                                                 'left_rear_limb_joint_1',
                                                 'left_rear_limb_joint_2'] -
                                           cur_q['right_front_limb_joint_1',
                                                 'right_front_limb_joint_2',
                                                 'right_rear_limb_joint_1',
                                                 'right_rear_limb_joint_2']))

        # mirror_enforce
        reward = 1 + horizontal_pos_rwd + horizontal_vel_rwd - rotate_pen - orth_pen - novelDiff / 4

        notdone = np.isfinite(ob[5::]).all() and (np.abs(angs) < np.pi / 2.0).all()

        done = not notdone

        return ob, reward, done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                  'horizontal_vel_rwd': horizontal_vel_rwd,
                                  'rotate_pen': -rotate_pen, 'orth_pen': -orth_pen, 'tau': tau,
                                  'symm_pos_pens': -symm_pos_pen, 'novelty': novelDiff}
