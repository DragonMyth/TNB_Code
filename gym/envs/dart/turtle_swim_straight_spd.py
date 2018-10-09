import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .simple_water_world import BaseFluidEnhancedSimulator
from .simple_water_world import BaseFluidEnhancedAllDirSimulator
from keras.models import load_model


class DartTurtleSwimStraighSPDEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 8, [-1.0] * 8])
        self.action_scale = np.pi / 2.0
        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'large_flipper_turtle_real.skel', self.frame_skip, 27, control_bounds, dt=0.002,
                                  disableViewer=True,
                                  custom_world=BaseFluidEnhancedAllDirSimulator)

        utils.EzPickle.__init__(self)
        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

        num_of_dofs = len(self.robot_skeleton.dofs) - len(self.robot_skeleton.joints[0].dofs)

        self.simulation_dt = self.dt * 1.0 / self.frame_skip
        self.Kp = np.diagflat([0.0] * len(self.robot_skeleton.joints[0].dofs) + [4000.0] * num_of_dofs)
        self.Kd = 120 * self.simulation_dt * self.Kp

        # self.Kd = 20 * self.simulation_dt * self.Kp
        # Symm rate Positive to be enforcing Symmetry
        #          Negative to be enforcing Asymmetry

        self.dqLim = 25  # This works for path finding model
        self.qLim = np.pi / 2

        self.stepNum = 0
        self.recordGap = 3

        init_obs = self._get_obs()
        self.novelty_window_size = 15
        self.traj_buffer = []  # [init_obs] * 5

        self.novel_autoencoders = []

        self.sum_of_old = 0
        self.sum_of_new = 0
        self.novelty_factor = 5

        self.novelDiff = 0
        self.novelDiffRev = 0
        self.path_data = []
        self.ret = 0
        self.ignore_obs = 11

    def _step(self, a):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        target_joint_pos = np.clip(a * self.action_scale, -self.action_scale, self.action_scale)

        target_pos = np.concatenate(([0.0] * 6, target_joint_pos))
        # SPD Controller

        for i in range(self.frame_skip):
            M = self.robot_skeleton.M + self.Kd * self.simulation_dt
            p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
            d = -self.Kd.dot(self.robot_skeleton.dq)

            qddot = np.linalg.solve(M, -self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
            tau = p + d - self.Kd.dot(qddot) * self.simulation_dt

            tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
            self.do_simulation(tau, 1)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        novelRwd, novelPenn = self.calc_novelty_from_autoencoder(ob)

        angs = np.abs(self.robot_skeleton.q[6::])
        old_angs = np.abs(old_q[6::])

        energy_rwd =0 #3 * sum(np.abs(old_angs - angs))

        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 1000

        orth_pen = 5 * (np.abs(cur_com[1] - self.original_com[1]) + np.abs(cur_com[2] - self.original_com[2]))
        rotate_pen = 5 * (np.abs(cur_q[0]) + np.abs(cur_q[1]) + np.abs(cur_q[2]))

        # mirror_enforce
        reward = 0 + horizontal_pos_rwd  - rotate_pen - orth_pen

        valid = np.isfinite(ob[self.ignore_obs::]).all()
        done = not valid

        return ob, (reward, -novelPenn*(5-reward)), done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                                'rotate_pen': -rotate_pen, 'orth_pen': -orth_pen, 'tau': tau,
                                                'energy_rwd': energy_rwd}

    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q[1:6], self.robot_skeleton.dq[0:6], self.robot_skeleton.q[6::],
                               self.robot_skeleton.dq[6::]]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        self.stepNum = 0
        self.sum_of_old = 0
        self.sum_of_new = 0
        self.ret = 0
        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(-60)
        self.track_skeleton_id = 0

    def normalizeTraj(self, traj, minq, maxq, mindq, maxdq):
        traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (maxq - minq)
        traj[:, :, int(len(traj[0, 0]) / 2)::] /= (maxdq - mindq)
        return traj

    def calc_novelty_from_autoencoder(self, obs):
        novelRwd = 0
        novelPenn = 0
        if len(self.novel_autoencoders) > 0:
            if (self.stepNum % self.recordGap == 0):
                if (np.isfinite(obs).all()):
                    self.traj_buffer.append(obs[self.ignore_obs:])

            if (len(self.traj_buffer) == self.novelty_window_size):
                novelDiffList = []
                # Reshape to 1 dimension
                traj_seg = np.array([self.traj_buffer])
                traj_seg = self.normalizeTraj(traj_seg, -self.qLim, self.qLim, -self.dqLim, self.dqLim)
                traj_seg = traj_seg.reshape((len(traj_seg), np.prod(traj_seg.shape[1:])))

                for i in range(len(self.novel_autoencoders)):
                    autoencoder = self.novel_autoencoders[i]

                    traj_recons = autoencoder.predict(traj_seg)
                    # if (self.stepNum == 20):
                    #     print("Original traj sum: ", np.sum(traj_seg))
                    #     print("Reconstructed traj sum: ", np.sum(traj_recons))

                    diff = traj_recons - traj_seg

                    normDiff = np.linalg.norm(diff[:], axis=1)[0]
                    novelDiffList.append(normDiff)
                    # print("Novel diff is", nov elDiff)
                self.traj_buffer.pop(0)

                self.novelDiff = min(novelDiffList)

                self.novelDiffRev = 4 - min(self.novelDiff, 4)

                self.sum_of_old += self.novelDiffRev
                self.sum_of_new += self.novelDiff
            novelRwd = self.novelty_factor * self.novelDiff
            novelPenn = self.novelty_factor * self.novelDiffRev
        return novelRwd, novelPenn
