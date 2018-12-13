import numpy as np
from gym import utils
from gym.envs.dart import dart_env


class DartWalker2dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0] * 6, [-1.0] * 6])
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])
        obs_dim = 17

        dart_env.DartEnv.__init__(self, 'walker2d.skel', 4, obs_dim, self.control_bounds, disableViewer=True)
        try:
            self.dart_world.set_collision_detector(3)
        except Exception as e:
            print('Does not have ODE collision detector, reverted to bullet collision detector')
            self.dart_world.set_collision_detector(2)
        utils.EzPickle.__init__(self)

        self.stepNum = 0
        self.recordGap = 3

        self.novelty_window_size = 15
        self.traj_buffer = []  # [init_obs] * 5

        self.novel_autoencoders = []

        self.sum_of_old = 0
        self.sum_of_new = 0
        self.novelty_factor = 2

        self.novelDiff = 0
        self.novelDiffRev = 0
        self.ignore_obs = 5

        self.normScale = self.generateNormScaleArr([6, np.pi / 2.0, 6, 10])

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

        pre_state = [self.state_vector()]

        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[3:] = clamped_control * self.action_scale
        posbefore = self.robot_skeleton.q[0]

        self.do_simulation(tau, self.frame_skip)

        ob = self._get_obs()
        novelRwd, novelPenn = self.calc_novelty_from_autoencoder(ob)

        posafter, ang = self.robot_skeleton.q[0, 2]
        height = self.robot_skeleton.bodynodes[2].com()[1]

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        alive_bonus = 1.0
        vel = (posafter - posbefore) / self.dt
        reward = vel
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()

        # uncomment to enable knee joint limit penalty
        '''joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        reward -= 5e-1 * joint_limit_penalty'''

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .8) and (height < 2.0) and (abs(ang) < 1.0))

        ob = self._get_obs()

        return ob, (reward, -novelPenn), done, {}

    def _get_obs(self):

        dq = np.clip(self.robot_skeleton.dq, -10, 10)
        state = np.concatenate([
            self.robot_skeleton.q[1:3], dq[0:3], self.robot_skeleton.q[3::], dq[3::]
        ])
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

        return state

    def reset_model(self):
        self.stepNum = 0
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5

    def normalizeTraj(self, traj):

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

                # self.sum_of_old += self.novelDiffRev
                # self.sum_of_new += self.novelDiff
            novelRwd = self.novelty_factor * self.novelDiff
            novelPenn = self.novelDiffRev
        return novelRwd, novelPenn
