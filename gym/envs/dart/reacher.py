# This environment is created by Karen Liu (karen.liu@gmail.com)

import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartReacherEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.target = np.array([0.8, -0.6, 0.6])
        self.action_scale = np.array([10, 10, 10, 10, 10])
        self.control_bounds = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0, -1.0]])
        dart_env.DartEnv.__init__(self, 'reacher.skel', 4, 26, self.control_bounds, disableViewer=True)
        utils.EzPickle.__init__(self)

        self.dqLim = 25
        self.qLim = np.pi

        self.stepNum = 0
        self.recordGap = 2

        self.novelty_window_size = 10
        self.traj_buffer = []  # [init_obs] * 5

        self.novel_autoencoders = []

        self.sum_of_old = 0
        self.sum_of_new = 0
        self.novelty_factor = 1

        self.novelDiff = 0
        self.novelDiffRev = 0
        self.path_data = []
        self.ret = 0
        self.ignore_obs = 6

        self.normScale = self.generateNormScaleArr([10, 1, 5, 2 * np.pi, 5, 50])

        self.longest_dist = 0

    def generateNormScaleArr(self, norm_scales):
        norms = np.zeros(len(self._get_obs()[self.ignore_obs::]))

        cur_idx = 0
        for i in range(0, len(norm_scales), 2):
            num_repeat = int(norm_scales[i])
            norms[cur_idx:cur_idx + num_repeat] = norm_scales[i + 1]
            cur_idx += num_repeat
        return norms

    def _step(self, a):
        self.stepNum += 1
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.multiply(clamped_control, self.action_scale)

        fingertip = np.array([0.0, -0.25, 0.0])
        vec = self.robot_skeleton.bodynodes[2].to_world(fingertip) - self.target

        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(tau).sum() * 0.001
        alive_bonus = 0

        reward = reward_dist + reward_ctrl + alive_bonus

        self.do_simulation(tau, self.frame_skip)

        ob = self._get_obs()

        novelRwd, novelPenn = self.calc_novelty_from_autoencoder(ob)

        s = self.state_vector()

        dist = np.linalg.norm(self.robot_skeleton.bodynodes[2].to_world(fingertip) - self.target)

        done = not (np.isfinite(s).all() and (-reward_dist > 0.1))

        #
        # if dist > self.longest_dist:
        #     done = True
        #     self.longest_dist = dist
        #
        # if done or self.stepNum == 500:
        #     print(self.longest_dist)
        return ob, (reward, -novelPenn), done, {'rwd': reward,
                                                'states': s, 'actions': tau,
                                                'NoveltyRwd': novelRwd, 'NoveltyPenn': -novelPenn
                                                }

    def _get_obs(self):
        theta = self.robot_skeleton.q
        fingertip = np.array([0.0, -0.25, 0.0])
        vec = self.robot_skeleton.bodynodes[2].to_world(fingertip) - self.target
        return np.concatenate(
            [self.target, vec, np.cos(theta), np.sin(theta), self.robot_skeleton.q, self.robot_skeleton.dq]).ravel()

    def reset_model(self):
        self.stepNum = 0

        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        # while True:
        #     self.target = self.np_random.uniform(low=-1, high=1, size=3)
        #     if np.linalg.norm(self.target) < 1.5: break

        self.dart_world.skeletons[0].q = [0, 0, 0, self.target[0], self.target[1], self.target[2]]

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0

    # def normalizeTraj(self, traj, minq, maxq, mindq, maxdq):
    #     traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (maxq - minq)
    #     traj[:, :, int(len(traj[0, 0]) / 2)::] /= (maxdq - mindq)
    #     return traj

    def normalizeTraj(self, traj):

        # traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (self.qnorm)
        # traj[:, :, int(len(traj[0, 0]) / 2)::] /= (self.dqnorm)

        traj[:, :, :] /= self.normScale
        return traj

    def calc_novelty_from_autoencoder(self, obs):
        novelRwd = 0
        novelPenn = 0
        if len(self.novel_autoencoders) > 0:
            if (self.stepNum % self.recordGap == 0):
                if (np.isfinite(obs).all()):
                    self.traj_buffer.append(obs[self.ignore_obs:])

            if (len(self.traj_buffer) == self.novelty_window_size):
                # print(self.stepNum)
                novelDiffList = []
                # Reshape to 1 dimension
                traj_seg = np.array([self.traj_buffer])
                traj_seg = self.normalizeTraj(traj_seg)
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

                self.novelDiffRev = np.exp(-self.novelDiff * self.novelty_factor)
                # self.novelDiffRev = 4 - min(self.novelDiff, 4)

                self.sum_of_old += self.novelDiffRev
                self.sum_of_new += self.novelDiff
            novelRwd = self.novelty_factor * self.novelDiff
            novelPenn = self.novelDiffRev
        return novelRwd, novelPenn
