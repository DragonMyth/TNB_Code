import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .simple_water_world import BaseFluidNoBackSimulator
from .simple_water_world import BaseFluidEnhancedAllDirSimulator
from .utils import *
from keras.models import load_model


class DartHumanoidSwimStraightFullDataCollectionEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 26, [-1.0] * 26])
        self.action_scale = np.pi / 2
        # self.action_scale = 5

        # self.torque_scale = 0.03
        self.frame_skip = 1
        dart_env.DartEnv.__init__(self, 'humanoid_swimmer_full.skel', self.frame_skip, 57, control_bounds, dt=0.002,
                                  disableViewer=True,
                                  custom_world=BaseFluidEnhancedAllDirSimulator)
        utils.EzPickle.__init__(self)

        self.init_nec_params()

    def init_nec_params(self):
        self.bodynodes_dict = construct_skel_dict(self.robot_skeleton.bodynodes)

        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

        num_of_dofs = len(self.robot_skeleton.dofs) - len(self.robot_skeleton.joints[0].dofs)
        #
        self.simulation_dt = self.dt  # * 1.0 / self.frame_skip
        # self.Kp = np.diagflat([0.0] * len(self.robot_skeleton.joints[0].dofs) + [4000.0] * num_of_dofs)
        # self.Kd = 80 * self.simulation_dt * self.Kp

        self.Kp = np.diagflat([0.0] * len(self.robot_skeleton.joints[0].dofs) + [20.0] * num_of_dofs)
        self.Kd = np.diagflat([0.0] * len(self.robot_skeleton.joints[0].dofs) + [2.0] * num_of_dofs)

        self.stepNum = 0
        self.recordGap = 3
        self.traj_buffer = []
        self.symm_autoencoder = None # load_model("./AutoEncoder/DartHumanFullAutoencoder_1+2+3+4.h5")
        self.novelty_factor = 20
        if self.symm_autoencoder is not None:
            print("Model name is", self.symm_autoencoder.name)

    def _step(self, a):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        target_pos = self.build_target_pos(a)

        # SPD Control
        #
        # for _ in range(self.frame_skip):
        #     M = self.robot_skeleton.M + self.Kd * self.simulation_dt
        #     p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
        #     d = -self.Kd.dot(self.robot_skeleton.dq)
        #
        #     qddot = np.linalg.solve (M, -self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        #     tau = p + d - self.Kd.dot(qddot) * self.simulation_dt
        #     tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
        #
        #     self.do_simulation(tau, 1)

        for _ in range(self.frame_skip):
            p = -self.Kp.dot(self.robot_skeleton.q - target_pos)
            d = -self.Kd.dot(self.robot_skeleton.dq)
            tau = p + d
            # tau[:] *= self.torque_scale
            #     tau = np.zeros(self.robot_skeleton.ndofs)
            #     tau[6:] = a * self.action_scale

            tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0

            self.do_simulation(tau, 1)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        novelDiff = 0
        if self.symm_autoencoder:
            if (self.stepNum % self.recordGap == 0):
                # 5 here is the num of dim for root related states
                self.traj_buffer.append(ob[9::])
            if (len(self.traj_buffer) == 15):
                # Reshape to 1 dimension
                traj_seg = np.array([self.traj_buffer])
                traj_seg = self.normalizeTraj(traj_seg, -np.pi, np.pi, -30, 30)
                traj_seg = traj_seg.reshape((len(traj_seg), np.prod(traj_seg.shape[1:])))

                traj_recons = self.symm_autoencoder.predict(traj_seg)
                diff = traj_recons - traj_seg
                novelDiff = np.linalg.norm(diff[:], axis=1)[0]
                # print("Novel diff is", nov elDiff)
                self.traj_buffer.pop(0)

            self.stepNum += 1

            novelDiff = self.novelty_factor * novelDiff

        angs = np.abs(self.robot_skeleton.q[6::])

        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 5000

        orth_pen = 0.5 * (np.abs(cur_com[1] - self.original_com[1]) + np.abs(cur_com[2] - self.original_com[2]))

        rotate_pen = np.sum(np.abs(cur_q[:3] - self.original_q[:3]))

        energy_consumed_pen = 0.5 * np.sum(np.abs(tau[8::] * old_dq[8::] * self.simulation_dt))

        # mirror_enforce
        reward = 1 + horizontal_pos_rwd - rotate_pen - orth_pen - energy_consumed_pen + novelDiff

        notdone = np.isfinite(ob[5::]).all() and (np.abs(angs) < np.pi / 2.0).all()
        done = not notdone

        return ob, reward, done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                  'rotate_pen': -rotate_pen, 'orth_pen': -orth_pen,
                                  'energy_consumed_pen': -energy_consumed_pen, 'tau': tau[8::], 'novelty': novelDiff}

    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q[4:8], self.robot_skeleton.dq[3:8], self.robot_skeleton.q[8::],
                               self.robot_skeleton.dq[8::]]).ravel()

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
        a = a * self.action_scale
        return np.concatenate(([0.0] * len(self.robot_skeleton.joints[0].dofs), a))

    def normalizeTraj(self, traj, minq, maxq, mindq, maxdq):
        traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (maxq - minq)
        traj[:, :, int(len(traj[0, 0]) / 2)::] /= (maxdq - mindq)
        return traj
