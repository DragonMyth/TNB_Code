import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from .simple_water_world import BaseFluidEnhancedAllDirSimulator
from keras.models import load_model


class DartCubePathFindingEnforceEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 2, [-1.0] * 2])

        self.action_scale = 15
        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'cube_path_finding.skel', self.frame_skip, 4, control_bounds, dt=0.005,
                                  disableViewer=True)

        utils.EzPickle.__init__(self)

        goal_skel = self.dart_world.skeletons[1]

        self.goal_com = goal_skel.C
        num_of_dofs = len(self.robot_skeleton.dofs) - len(self.robot_skeleton.joints[0].dofs)

        self.simulation_dt = self.dt * 1.0 / self.frame_skip

        self.stepNum = 0
        self.recordGap = 3
        self.traj_buffer = []
        self.symm_autoencoder = None  # load_model("./AutoEncoder/local/path_finding_autoencoder_1+2+3+4.h5")
        self.novelty_factor = 200
        self.progress_ratio = 0
        print("Initial progress ratio: ", self.progress_ratio)

    def _step(self, a):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        old_to_goal_dist = (old_com[0] - self.goal_com[0]) ** 2 + (old_com[2] - self.goal_com[2]) ** 2

        tau = np.concatenate((a * self.action_scale, [0.0]))
        self.do_simulation(tau, self.frame_skip)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        cur_to_goal_dist = (cur_com[0] - self.goal_com[0]) ** 2 + (cur_com[2] - self.goal_com[2]) ** 2

        # this should be the unscaled novelty term
        novelDiff = 0
        if self.symm_autoencoder is not None:
            if (self.stepNum % self.recordGap == 0):
                # 5 here is the num of dim for root related states
                if (np.isfinite(ob).all()):
                    self.traj_buffer.append(ob[:])
            if (len(self.traj_buffer) == 15):
                # Reshape to 1 dimension
                traj_seg = np.array([self.traj_buffer])
                traj_seg = self.normalizeTraj(traj_seg, -np.pi, np.pi, -5, 5)
                traj_seg = traj_seg.reshape((len(traj_seg), np.prod(traj_seg.shape[1:])))

                traj_recons = self.symm_autoencoder.predict(traj_seg)
                diff = traj_recons - traj_seg
                novelDiff = np.linalg.norm(diff[:], axis=1)[0]
                # print("Novel diff is", nov elDiff)
                self.traj_buffer.pop(0)

            self.stepNum += 1

        novelDiff = self.novelty_factor * novelDiff

        # print((1 - self.progress_ratio))
        # print(novelDiff)
        #
        # print("Curr Box Position: ",cur_com)
        # print("Distance to Goal: ",cur_to_goal_dist)
        distant_rwd = -30 + 200 * (old_to_goal_dist - cur_to_goal_dist) + min(140, 100 / cur_to_goal_dist)
        # mirror_enforce
        reward = novelDiff
        if (np.isfinite(distant_rwd)):
            reward += distant_rwd
        else:
            reward += -2500

        valid = np.isfinite(ob[:]).all()

        if np.sqrt(cur_to_goal_dist) <= 0.2:
            reward += 10000
            valid = False
        done = not valid

        return ob, reward, done, {'Total Reward': reward, 'distant_rwd': distant_rwd, 'tau': tau, 'novelty': novelDiff}

    def _get_obs(self):
        return np.concatenate(
            [[self.robot_skeleton.C[0]], [self.robot_skeleton.C[2]], self.robot_skeleton.dq[0:2]]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[0] = 0
        self._get_viewer().scene.tb.trans[0] = 0

        self._get_viewer().scene.tb.trans[2] = -6.5
        self._get_viewer().scene.tb._set_theta(-90)
        self.track_skeleton_id = 0

    def normalizeTraj(self, traj, minq, maxq, mindq, maxdq):
        traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (maxq - minq)
        traj[:, :, int(len(traj[0, 0]) / 2)::] /= (maxdq - mindq)
        return traj


class DartCubePathFindingReleaseEnv(DartCubePathFindingEnforceEnv):
    def __init__(self):
        DartCubePathFindingEnforceEnv.__init__(self)
        # self.symm_autoencoder = None

    def _step(self, a):

        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        old_to_goal_dist = (old_com[0] - self.goal_com[0]) ** 2 + (old_com[2] - self.goal_com[2]) ** 2

        tau = np.concatenate((a * self.action_scale, [0.0]))
        self.do_simulation(tau, self.frame_skip)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        cur_to_goal_dist = (cur_com[0] - self.goal_com[0]) ** 2 + (cur_com[2] - self.goal_com[2]) ** 2

        # this should be the unscaled novelty term
        novelDiff = 0
        if self.symm_autoencoder is not None:
            if (self.stepNum % self.recordGap == 0):
                # 5 here is the num of dim for root related states
                if (np.isfinite(ob).all()):
                    self.traj_buffer.append(ob[:])
            if (len(self.traj_buffer) == 15):
                # Reshape to 1 dimension
                traj_seg = np.array([self.traj_buffer])
                traj_seg = self.normalizeTraj(traj_seg, -np.pi, np.pi, -5, 5)
                traj_seg = traj_seg.reshape((len(traj_seg), np.prod(traj_seg.shape[1:])))

                traj_recons = self.symm_autoencoder.predict(traj_seg)
                diff = traj_recons - traj_seg
                novelDiff = np.linalg.norm(diff[:], axis=1)[0]
                # print("Novel diff is", nov elDiff)
                self.traj_buffer.pop(0)

            self.stepNum += 1

        novelDiff = self.novelty_factor * novelDiff

        # print((1 - self.progress_ratio))
        # print(novelDiff)
        #
        # print("Curr Box Position: ",cur_com)
        # print("Distance to Goal: ",cur_to_goal_dist)

        distant_rwd = 200 * (old_to_goal_dist - cur_to_goal_dist)

        reward = 0
        if (novelDiff < 70):
            reward = - 0.1 * (70 - novelDiff)

        if (np.isfinite(distant_rwd)):
            reward += distant_rwd
        else:
            reward += -500

        valid = np.isfinite(ob[:]).all()

        if np.sqrt(cur_to_goal_dist) <= 0.1:
            reward += 5500
            valid = False
        done = not valid

        return ob, reward, done, {'rwd': reward, 'horizontal_pos_rwd': distant_rwd,
                                  'rotate_pen': -0, 'orth_pen': -0, 'tau': tau,
                                  'symm_pos_pens': -0, 'novelty': novelDiff,
                                  'energy_consumed_pen': -0}
