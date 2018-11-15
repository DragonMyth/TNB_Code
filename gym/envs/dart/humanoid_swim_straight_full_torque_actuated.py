import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .simple_water_world import BaseFluidNoBackSimulator
from .simple_water_world import BaseFluidEnhancedAllDirSimulator
from .utils import *
from keras.models import load_model


class DartHumanoidSwimStraightFullTorqueActuateEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 20, [-1.0] * 20])
        self.action_scale = np.array([10] * 20)
        # self.action_scale = 5

        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'humanoid_swimmer_full_reduce_dim.skel', self.frame_skip, 51, control_bounds,
                                  dt=0.002,
                                  disableViewer=True,
                                  custom_world=BaseFluidEnhancedAllDirSimulator)
        utils.EzPickle.__init__(self)

        self.bodynodes_dict = construct_skel_dict(self.robot_skeleton.bodynodes)

        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

        num_of_dofs = len(self.robot_skeleton.dofs) - len(self.robot_skeleton.joints[0].dofs)
        #
        # self.Kp = np.diagflat([0.0] * len(self.robot_skeleton.joints[0].dofs) + [4000.0] * num_of_dofs)
        # self.Kd = 80 * self.simulation_dt * self.Kp

        self.stepNum = 0
        self.recordGap = 3

        self.traj_buffer = []
        self.novel_autoencoders = []
        self.novelty_factor = 5
        self.novelDiff = 0
        self.novelDiffRev = 0
        self.ignore_obs = 11
        self.normScale = self.generateNormScaleArr([1, 1])
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 30
        }
        # q = self.robot_skeleton.q
        # q[3] = 10
        # self.robot_skeleton.set_positions(q)

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

        self.do_simulation(tau, self.frame_skip)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        novelRwd, novelPenn = self.calc_novelty_from_autoencoder(ob)

        angs = np.abs(self.robot_skeleton.q[6::])

        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 1000

        orth_pen = 1 * (np.abs(cur_com[1] - self.original_com[1]) + np.abs(cur_com[2] - self.original_com[2]))

        rotate_pen = np.sum(np.abs(cur_q[:3] - self.original_q[:3]))

        # energy_consumed_pen = 0.5 * np.sum(np.abs(tau[8::] * old_dq[8::] * self.simulation_dt))

        # mirror_enforce
        reward = horizontal_pos_rwd - rotate_pen - orth_pen

        valid = np.isfinite(ob[5::]).all()
        done = not valid

        return ob, (reward, -novelPenn), done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                                'rotate_pen': -rotate_pen, 'orth_pen': -orth_pen, 'actions': tau[6::],
                                                'states': ob[self.ignore_obs::],
                                                'NoveltyRwd': novelRwd, 'NoveltyPenn': -novelPenn
                                                }

    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q[0:3],self.robot_skeleton.q[4:6], self.robot_skeleton.dq[0:6],
                               self.robot_skeleton.q[6:8], self.robot_skeleton.dq[6:8],
                               self.robot_skeleton.q[8::], self.robot_skeleton.dq[8::]]).ravel()

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

                    diff = diff.reshape(self.novelty_window_size, len(obs[self.ignore_obs:]))
                    diff = np.multiply(diff, self.novelty_weight_mat)

                    diff = diff.reshape((len(diff), np.prod(diff.shape[1:])))

                    normDiff = np.linalg.norm(diff[:], axis=1)[0]
                    novelDiffList.append(normDiff)
                    # print("Novel diff is", nov elDiff)
                self.traj_buffer.pop(0)

                self.novelDiff = min(novelDiffList)

                self.novelDiffRev = np.exp(-self.novelDiff)
                # self.novelDiffRev = 4 - min(self.novelDiff, 4)

            novelRwd = self.novelty_factor * self.novelDiff
            novelPenn = self.novelty_factor * self.novelDiffRev
        return novelRwd, novelPenn
