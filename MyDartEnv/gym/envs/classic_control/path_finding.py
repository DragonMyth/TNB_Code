"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from keras.models import load_model
import joblib

logger = logging.getLogger(__name__)


class PathFinding(gym.Env):
    def __init__(self):

        self.grid_map = np.zeros((21, 21), dtype=int)
        self.grid_map[0, :] = 1
        self.grid_map[-1, :] = 1
        self.grid_map[:, 0] = 1
        self.grid_map[:, -1] = 1

        self.grid_map[4, 4:-7] = 1
        self.grid_map[4:-7, 4] = 1
        self.grid_map[-5, 7:-4] = 1
        self.grid_map[7:-4, -5] = 1
        self.grid_map[8:13, 8:13] = 1

        # This is the goal grid
        goal_i, goal_j = 2, 18
        self.grid_map[goal_i, goal_j] = 2

        self.grid_size = 0.25
        self.grid_vis_size = 30

        self.point_mass = 10
        self.point_pos = -2 * np.ones(2)
        self.point_vel = -np.zeros(2)
        self.point_acc_force = -np.zeros(2)

        self.original_pos = self.point_pos

        self.goal_pos = self.grid_idx_to_pos(goal_i, goal_j)

        self.dt = 0.002
        self.frameskip = 5

        self.action_scale = 10
        self.obs_dim = 6

        self.action_dim = 2
        action_high = np.ones(self.action_dim)
        action_low = -action_high
        self.action_space = spaces.Box(action_low, action_high)
        # self.action_space = None
        obs_high = np.inf * np.ones(self.obs_dim)
        obs_low = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high)

        self._seed()
        self.viewer = None
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt)) / self.frameskip
        }

        self.stepNum = 0
        self.recordGap = 3

        init_obs = self._get_obs()
        self.traj_buffer = [init_obs] * 15

        self.novel_autoencoders = []
        self.novel_visitations = []

        autoencoder_dir = "./AutoEncoder/local/"
        # autoencoder1 = load_model(autoencoder_dir + "new_path_finding_biased_autoencoder_1.h5")
        # autoencoder2 = load_model(autoencoder_dir + "new_path_finding_biased_autoencoder_2.h5")
        # autoencoder3 = load_model(autoencoder_dir + "new_path_finding_biased_autoencoder_3.h5")
        # autoencoder4 = load_model(autoencoder_dir + "new_path_finding_autoencoder_4.h5")

        # self.novel_autoencoders.append(autoencoder1)
        # self.novel_autoencoders.append(autoencoder2)
        # self.novel_autoencoders.append(autoencoder3)
        # self.novel_autoencoders.append(autoencoder4)
        print("Autoencoder Model names: ")
        for i in range(len(self.novel_autoencoders)):
            print("Model {} name is {}".format(i, self.novel_autoencoders[i].name))

        visitation_dir = '../novelty_data/local/visitation_grid/'

        # visitation_grid_1 = joblib.load(visitation_dir + 'fill_bottom_lane.pkl')
        # visitation_grid_2 = joblib.load(visitation_dir + 'fill_second_bottom_lane.pkl')
        # visitation_grid_3 = joblib.load(visitation_dir + 'fill_third_bottom_lane.pkl')
        # visitation_grid_4 = joblib.load(visitation_dir + 'fill_fourth_bottom_lane.pkl')

        # self.novel_visitations.append(visitation_grid_1)
        # self.novel_visitations.append(visitation_grid_2)
        # #
        # self.novel_visitations.append(visitation_grid_3)
        # self.novel_visitations.append(visitation_grid_4)

        self.novelty_factor = 100
        self.novelDiff = 0
        self.sum_of_old = 0

        # self.novelty_discount = 1
        # self.novelty_discount_rate = 0.99
        self.path_data = []

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # print(action)
        pos_before = self.point_pos

        old_to_goal_dist = (pos_before[0] - self.goal_pos[0]) ** 2 + (pos_before[1] - self.goal_pos[1]) ** 2

        tau = action * self.action_scale

        wall_hit = self.do_simulation(tau, self.frameskip)

        obs = self._get_obs()

        pos_after = self.point_pos

        curr_to_goal_dist = (pos_after[0] - self.goal_pos[0]) ** 2 + (pos_after[1] - self.goal_pos[1]) ** 2
        #
        if (len(self.path_data) == 2):
            self.path_data.pop(0)

        self.path_data.append([pos_after, -1])

        self.stepNum += 1

        novelRwd = self.calc_novelty_from_visitation(obs)

        self.sum_of_old += self.novelDiff

        if (self.novelDiff > 0):
            self.path_data[-1][1] = self.novelDiff

        i, j = self.pos_to_grid_idx(pos_after)

        close_to_wall_penalty = 0
        for neighbor_c in range(i - 2, i + 3, 1):
            for neighbor_r in range(j - 2, j + 3, 1):
                if neighbor_c > 0 and neighbor_c < len(self.grid_map) and neighbor_r > 0 and neighbor_r < len(
                        self.grid_map[0]):
                    grid_val = self.grid_map[neighbor_c, neighbor_r]
                    if grid_val == 1:
                        wall_pos = self.grid_idx_to_pos(neighbor_c, neighbor_r)
                        dist_sq = (pos_after[0] - wall_pos[0]) ** 2 + (pos_after[1] - wall_pos[1]) ** 2
                        if (close_to_wall_penalty < 1.0 / dist_sq):
                            close_to_wall_penalty = 1 / dist_sq

        alive_penalty = -10
        progress_reward = 100 * (old_to_goal_dist - curr_to_goal_dist)
        distance_to_goal_reward = min(50, 10 / curr_to_goal_dist)
        close_to_wall_penalty = min(10, close_to_wall_penalty)

        reward = -novelRwd + alive_penalty + progress_reward + distance_to_goal_reward - close_to_wall_penalty

        done = False

        if self.grid_map[i, j] == 2:
            done = True
            reward += 5000
        if wall_hit:
            reward -= 2000
            done = True

        if self.sum_of_old > 20:
            reward -= 5000
            done = True

        # reward = 0.1 * reward + novelRwd

        return obs, reward, done, {'Alive penalty': alive_penalty,
                                   'Progress Reward': progress_reward,
                                   'Distance to Goal Reward': distance_to_goal_reward, 'tau': tau, 'Novelty': novelRwd,
                                   'Total Reward': reward, 'Close to Wall Penalty'
                                   : -close_to_wall_penalty}

    def _get_obs(self):
        return np.concatenate([[self.point_pos[0], self.point_pos[1]], [self.point_vel[0], self.point_vel[1]],
                               [self.goal_pos[0] - self.point_pos[0],
                                self.goal_pos[1] - self.point_pos[1]]]).ravel()

    def do_simulation(self, tau, frameskip):

        wall_hit = 0
        for _ in range(frameskip):
            self.point_acc_force = tau
            i_before, j_before = self.pos_to_grid_idx(self.point_pos)
            next_pos = self.point_pos + self.dt * self.point_vel
            i_after, j_after = self.pos_to_grid_idx(next_pos)

            # print(i_after, j_after)

            if self.grid_map[i_after, j_after] == 1:
                change_dir = np.array([abs(j_after - j_before), abs(i_after - i_before)])

                vel = np.zeros(len(self.point_vel))
                if (change_dir[0] == 1):
                    self.point_vel[0] *= -.2
                if (change_dir[1] == 1):
                    self.point_vel[1] *= -.2
                wall_hit += 1

            self.point_pos = self.point_pos + self.dt * self.point_vel
            self.point_vel = self.point_vel + self.dt * (self.point_acc_force / self.point_mass)
        return wall_hit

    def _reset(self):

        self.point_pos = -2 * np.ones(2) + self.np_random.uniform(low=-0.01, high=0.01, size=(2))

        self.point_vel = -np.zeros(2) + self.np_random.uniform(low=-0.01, high=0.01, size=(2))
        self.point_acc_force = -np.zeros(2)
        self.stepNum = 0
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            for i in range(len(self.grid_map)):
                for j in range(len(self.grid_map[0])):

                    l = screen_width / 2 + (
                            j - int((len(
                        self.grid_map)) / 2)) * self.grid_vis_size - self.grid_vis_size / 2

                    r = screen_width / 2 + (
                            j - int((len(
                        self.grid_map)) / 2)) * self.grid_vis_size + self.grid_vis_size / 2

                    t = screen_height / 2 + (
                            int((len(
                                self.grid_map)) / 2) - i) * self.grid_vis_size + self.grid_vis_size / 2
                    b = screen_height / 2 + (
                            int((len(
                                self.grid_map)) / 2) - i) * self.grid_vis_size - self.grid_vis_size / 2

                    cell = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

                    # cell.add_attr(rendering.Transform())

                    if (self.grid_map[i, j] == 1):
                        cell.set_color(0, 0, 0)

                    elif (self.grid_map[i, j] == 2):
                        cell.set_color(1, 0, 0)
                    else:
                        cell.set_color(1, 1, 1)

                    # right_edge = rendering.Line((l, b), (l, t))
                    right_edge = rendering.FilledPolygon([(l - 1, b), (l - 1, t), (l + 1, t), (l + 1, b)])

                    bottom_edge = rendering.FilledPolygon([(l, b - 1), (l, b + 1), (r, b + 1), (r, b - 1)])

                    right_edge.set_color(.3, .3, .3)
                    bottom_edge.set_color(.3, .3, .3)

                    self.viewer.add_geom(right_edge)

                    self.viewer.add_geom(cell)

                    self.viewer.add_geom(bottom_edge)

            q = self.point_pos

            q_x = screen_width / 2

            q_y = screen_height / 2

            point_mass = rendering.FilledPolygon(
                [(q_x - 7, q_y - 7), (q_x - 7, q_y + 7), (q_x + 7, q_y + 7), (q_x + 7, q_y - 7)])
            point_mass.set_color(0, 0, 1)
            self.point_mass_trans = rendering.Transform()
            point_mass.add_attr(self.point_mass_trans)

            self.viewer.add_geom(point_mass)

        q = self.point_pos

        q_x = (q[0] / self.grid_size * self.grid_vis_size)
        q_y = (q[1] / self.grid_size * self.grid_vis_size)

        # print(q)
        # idx,idy = self.pos_to_grid_idx(q)
        # print(idx,idy)
        # posx,posy = self.grid_idx_to_pos(idx,idy)
        # print(posx,posy)

        self.point_mass_trans.set_translation(q_x, q_y)

        if len(self.path_data) == 2:
            point_1 = self.path_data[0][0]
            point_2 = self.path_data[1][0]

            color_1 = self.path_data[0][1]
            color_2 = self.path_data[1][1]
            avg_color = ((color_1 + color_2) / 2.0)

            point_1_x = screen_width / 2 + (point_1[0] / self.grid_size * self.grid_vis_size)

            point_1_y = screen_height / 2 + (point_1[1] / self.grid_size * self.grid_vis_size)

            point_2_x = screen_width / 2 + (point_2[0] / self.grid_size * self.grid_vis_size)

            point_2_y = screen_height / 2 + (point_2[1] / self.grid_size * self.grid_vis_size)

            path_segment = rendering.FilledPolygon(
                [(point_1_x - 4, point_1_y - 4), (point_1_x - 4, point_1_y + 4), (point_1_x + 4, point_1_y + 4),
                 (point_1_x + 4, point_1_y - 4)])
            # print(avg_color)
            if (avg_color < 0):

                path_segment.set_color(1, 0, 1)
            else:
                path_segment.set_color(0, avg_color * 0.8, 0)

            self.viewer.add_geom(path_segment)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def pos_to_grid_idx(self, pos):

        normalized_pos = (np.round((pos) / (self.grid_size))).astype(int)

        x_idx = normalized_pos[0] + int(len(self.grid_map) / 2)
        y_idx = -normalized_pos[1] + int(len(self.grid_map) / 2)

        return y_idx, x_idx

    def grid_idx_to_pos(self, i, j):

        x = (j - int((len(self.grid_map)) / 2)) * self.grid_size

        y = (int((len(self.grid_map)) / 2) - i) * self.grid_size

        return np.array([x, y])

    def normalizeTraj(self, traj, minq, maxq, mindq, maxdq):
        traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (maxq - minq)
        traj[:, :, int(len(traj[0, 0]) / 2)::] /= (maxdq - mindq)
        return traj

    def calc_novelty_from_visitation(self, obs):
        novelRwd = 0

        novelDiffList = []
        for i in range(len(self.novel_visitations)):
            visitation_grid = self.novel_visitations[i]

            idx_i, idx_j = self.pos_to_grid_idx(np.array([obs[0], obs[1]]))
            novelDiffList.append(visitation_grid[idx_i, idx_j])

        if (len(novelDiffList) > 0):
            self.novelDiff = np.max(novelDiffList)
        novelRwd = self.novelDiff * self.novelty_factor
        return novelRwd

    def calc_novelty_from_path(self, obs):
        novelRwd = 0
        # if (self.old_path_1):
        diff = np.inf
        for i in range(len(self.novel_paths)):
            old_path = self.novel_paths[i]
            if (self.stepNum < len(old_path)):
                path_observation = old_path[self.stepNum]

                curr_diff = (obs[0] - path_observation[0]) ** 2 + (obs[1] - path_observation[1]) ** 2
                if (curr_diff < diff):
                    diff = curr_diff

            else:
                diff -= 1
        self.novelDiff = diff
        # print(self.novelDiff)
        novelRwd = self.novelty_factor * self.novelDiff
        novelRwd = min(20, novelRwd)
        return novelRwd

    def calc_novelty_from_autoencoder(self, obs):
        novelRwd = 0
        if len(self.novel_autoencoders) > 0:
            if (self.stepNum % self.recordGap == 0):
                # 5 here is the num of dim for root related states
                if (np.isfinite(obs).all()):
                    self.traj_buffer.append(obs[:])
            if (len(self.traj_buffer) == 15):
                novelDiffList = []
                # Reshape to 1 dimension
                traj_seg = np.array([self.traj_buffer])
                traj_seg = self.normalizeTraj(traj_seg, -10, 10, -10, 10)
                traj_seg = traj_seg.reshape((len(traj_seg), np.prod(traj_seg.shape[1:])))
                for i in range(len(self.novel_autoencoders)):
                    autoencoder = self.novel_autoencoders[i]
                    traj_recons = autoencoder.predict(traj_seg)
                    diff = traj_recons - traj_seg

                    normDiff = np.linalg.norm(diff[:], axis=1)[0]
                    novelDiffList.append(normDiff)
                    # print("Novel diff is", nov elDiff)
                self.traj_buffer.pop(0)

                self.novelDiff = min(novelDiffList)
                # print(self.novelDiff)

            if self.novelDiff < 0.06:
                self.novelDiff = 0
            else:
                self.novelDiff = min(self.novelDiff, 2)

            novelRwd = self.novelty_factor * self.novelDiff
        return novelRwd


class PathFindingReleasing(PathFinding):
    def __init__(self):

        PathFinding.__init__(self)
        self.symm_autoencoder = None

    def _step(self, action):

        pos_before = self.point_pos

        old_to_goal_dist = (pos_before[0] - self.goal_pos[0]) ** 2 + (pos_before[1] - self.goal_pos[1]) ** 2

        tau = action * self.action_scale

        wall_hit = self.do_simulation(tau, self.frameskip)

        obs = self._get_obs()

        pos_after = self.point_pos

        curr_to_goal_dist = (pos_after[0] - self.goal_pos[0]) ** 2 + (pos_after[1] - self.goal_pos[1]) ** 2

        novelRwd = 0
        if self.symm_autoencoder is not None:
            if (self.stepNum % self.recordGap == 0):
                # 5 here is the num of dim for root related states
                if (np.isfinite(obs).all()):
                    self.traj_buffer.append(obs[:])
            if (len(self.traj_buffer) == 15):
                # Reshape to 1 dimension
                traj_seg = np.array([self.traj_buffer])
                traj_seg = self.normalizeTraj(traj_seg, -3, 3, -4, 4)
                traj_seg = traj_seg.reshape((len(traj_seg), np.prod(traj_seg.shape[1:])))
                traj_recons = self.symm_autoencoder.predict(traj_seg)
                diff = traj_recons - traj_seg
                self.novelDiff = np.linalg.norm(diff[:], axis=1)[0]
                # print("Novel diff is", nov elDiff)
                self.traj_buffer.pop(0)

            self.stepNum += 1
            #
            # if self.novelDiff < 0.07:
            #     self.novelDiff = 0
            # else:
            #     self.novelDiff = min(self.novelDiff, 2)

            novelRwd = self.novelty_factor * self.novelDiff

        alive_penalty = -20
        progress_reward = 180 * (old_to_goal_dist - curr_to_goal_dist)
        distance_to_goal_reward = min(50, 10 / curr_to_goal_dist)

        i, j = self.pos_to_grid_idx(pos_after)

        close_to_wall_penalty = 0
        for neighbor_c in range(i - 2, i + 3, 1):
            for neighbor_r in range(j - 2, j + 3, 1):
                if neighbor_c > 0 and neighbor_c < len(self.grid_map) and neighbor_r > 0 and neighbor_r < len(
                        self.grid_map[0]):
                    grid_val = self.grid_map[neighbor_c, neighbor_r]
                    if grid_val == 1:
                        wall_pos = self.grid_idx_to_pos(neighbor_c, neighbor_r)
                        dist_sq = (pos_after[0] - wall_pos[0]) ** 2 + (pos_after[1] - wall_pos[1]) ** 2
                        if (close_to_wall_penalty < 1.0 / dist_sq):
                            close_to_wall_penalty = 1 / dist_sq

        close_to_wall_penalty = min(10, close_to_wall_penalty)
        reward = alive_penalty + progress_reward + distance_to_goal_reward - close_to_wall_penalty

        done = False
        if wall_hit == 1:
            reward -= 500
        elif self.grid_map[i, j] == 2:
            done = True
            reward += 20000

        return obs, reward, done, {'Alive penalty': alive_penalty,
                                   'Progress Reward': progress_reward,
                                   'Distance to Goal Reward': distance_to_goal_reward, 'tau': tau, 'Novelty': novelRwd,
                                   'Unscaled Novelty': self.novelDiff, 'Total Reward': reward, 'Close to Wall Penalty'
                                   : -close_to_wall_penalty}
