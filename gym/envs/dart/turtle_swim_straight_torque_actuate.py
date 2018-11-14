import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .simple_water_world import BaseFluidEnhancedSimulator
from .simple_water_world import BaseFluidEnhancedAllDirSimulator
from keras.models import load_model


class DartTurtleSwimStraighTorqueActuateEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 8, [-1.0] * 8])
        # self.action_scale = np.array([7.0, 7.0, 0.1, 0.1, 7.0, 7.0, 0.1, 0.1])  # np.pi / 2.0
        self.action_scale = np.array([7.0, 7.0, 0.04, 0.04, 7.0, 7.0, 0.04, 0.04])  # np.pi / 2.0
        self.frame_skip = 5

        dart_env.DartEnv.__init__(self, 'large_flipper_turtle_real.skel', self.frame_skip, 27, control_bounds, dt=0.002,
                                  disableViewer=True,
                                  custom_world=BaseFluidEnhancedAllDirSimulator)

        utils.EzPickle.__init__(self)
        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

        num_of_dofs = len(self.robot_skeleton.dofs) - 6

        self.simulation_dt = self.dt * 1.0 / self.frame_skip
        # self.dqLim = 25
        # self.qLim = np.pi / 2

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

        self.novelty_weight_mat = np.ones((self.novelty_window_size, len(init_obs[self.ignore_obs:])))

        # indexes = np.array([np.arange(2, 4), np.arange(6, 8), np.arange(10, 12), np.arange(14, 16)])

        # self.novelty_weight_mat[:, indexes] = 0

        self.normScale = self.generateNormScaleArr([8, 2 * np.pi, 8, 50])

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 30
        }

    def generateNormScaleArr(self, norm_scales):
        norms = np.zeros(len(self._get_obs()[self.ignore_obs::]))

        cur_idx = 0
        for i in range(0, len(norm_scales), 2):
            num_repeat = int(norm_scales[i])
            norms[cur_idx:cur_idx + num_repeat] = norm_scales[i + 1]
            cur_idx += num_repeat
        return norms

    def _step(self, a):

        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        tau = np.clip(a * self.action_scale, -self.action_scale, self.action_scale)

        tau = np.concatenate(([0.0] * 6, tau))
        # SPD Controller
        prev_obs = self._get_obs()
        self.do_simulation(tau, self.frame_skip)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        novelRwd, novelPenn = self.calc_novelty_from_autoencoder(ob)

        angs = np.abs(self.robot_skeleton.q[6::])
        old_angs = np.abs(old_q[6::])

        # energy_rwd = sum(np.abs(old_angs - angs))

        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 1000

        orth_pen = 1 * (np.abs(cur_com[1] - self.original_com[1]) + np.abs(cur_com[2] - self.original_com[2]))
        rotate_pen = 0.5 * (np.abs(cur_q[3]) + np.abs(cur_q[4]) + np.abs(cur_q[5]))

        # mirror_enforce
        reward = 0 + horizontal_pos_rwd - rotate_pen - orth_pen

        if (np.isnan(reward)):
            print('Horizontal', horizontal_pos_rwd)
            print('Orth', orth_pen)
            print('rotate', rotate_pen)
            print('Obs', ob)
            print('PrevObs', prev_obs)
            print('ObsFinite?', np.isfinite(ob[:]).all())
            print('Action', tau)
            reward = 0
            novelPenn = 1
        # print(reward)

        valid = np.isfinite(ob[:]).all() and (ob < 10e3).all()

        done = not valid
        self.stepNum += 1

        return ob, (reward, -novelPenn), done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                                'rotate_pen': -rotate_pen, 'orth_pen': -orth_pen, 'actions': tau[6::],
                                                'states': ob[self.ignore_obs::],
                                                'NoveltyRwd': novelRwd, 'NoveltyPenn': -novelPenn
                                                }

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
                    # diff = diff.reshape(self.novelty_window_size, len(obs[self.ignore_obs:]))
                    # diff = np.multiply(diff, self.novelty_weight_mat)

                    normDiff = np.linalg.norm(diff[:], axis=1)[0]
                    novelDiffList.append(normDiff)
                    # print("Novel diff is", nov elDiff)
                self.traj_buffer.pop(0)

                self.novelDiff = min(novelDiffList)

                self.novelDiffRev = np.exp(-self.novelDiff)
                # self.novelDiffRev = 4 - min(self.novelDiff, 4)
                self.sum_of_old += self.novelDiffRev
                self.sum_of_new += self.novelDiff
            novelRwd = self.novelty_factor * self.novelDiff
            novelPenn = self.novelty_factor * self.novelDiffRev
        return novelRwd, novelPenn
