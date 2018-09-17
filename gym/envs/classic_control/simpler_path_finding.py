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


class SimplerPathFinding(gym.Env):
    def __init__(self):

        # 0 for path
        # 1 for wall
        # 2,3,4,5 for goal
        # 6,7,8,9 for hints
        self.grid_map = np.zeros((27, 27), dtype=int)
        self.grid_map[0, :] = 1
        self.grid_map[-1, :] = 1
        self.grid_map[:, 0] = 1
        self.grid_map[:, -1] = 1

        mid_idx = int(len(self.grid_map) / 2)

        # self.grid_map[(int(len(self.grid_map) / 2)), 4:-4] = 1
        # self.grid_map[(int(len(self.grid_map) / 4)), 4:-4] = 1
        # self.grid_map[(int(len(self.grid_map) / 4)) * 3, 4:-4] = 1

        self.grid_map[1:mid_idx - 1, 1:mid_idx - 1] = 1
        self.grid_map[(mid_idx + 2):-1, 1:(mid_idx - 1)] = 1

        self.grid_map[1:mid_idx - 1, (mid_idx + 2):-1] = 1
        self.grid_map[(mid_idx + 2):-1, (mid_idx + 2):-1] = 1

        self.grid_map[mid_idx - 3:mid_idx - 1, mid_idx - 3: mid_idx - 1] = 0
        self.grid_map[mid_idx - 3:mid_idx - 1, mid_idx + 2: mid_idx + 4] = 0
        self.grid_map[mid_idx + 2: mid_idx + 4, mid_idx + 2: mid_idx + 4] = 0
        self.grid_map[mid_idx + 2: mid_idx + 4, mid_idx - 3: mid_idx - 1] = 0

        self.grid_map[mid_idx - 1:mid_idx + 2, 1] = 2
        self.grid_map[mid_idx - 1:mid_idx + 2, -2] = 3
        self.grid_map[1, mid_idx - 1:mid_idx + 2] = 4
        self.grid_map[-2, mid_idx - 1:mid_idx + 2] = 5

        self.grid_map[mid_idx, 2:mid_idx] = 6
        self.grid_map[mid_idx, mid_idx + 1:-2] = 7
        self.grid_map[2:mid_idx, mid_idx] = 8
        self.grid_map[mid_idx + 1:-2, mid_idx] = 9

        # This is the goal grid
        # self.grid_map[1:-1, -2] = 2

        self.grid_size = 0.25
        self.grid_vis_size = 25

        self.point_mass = 10

        self.init_pos = np.zeros(2)  # np.array([-2, -2])
        self.point_pos = self.init_pos

        self.point_vel = -np.zeros(2)
        self.point_acc_force = -np.zeros(2)

        self.original_pos = self.point_pos
        self.dt = 0.002
        self.frameskip = 5

        self.action_scale = 4
        self.obs_dim = 4

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

        self.dqLim = 5  # This works for path finding model
        self.qLim = 5
        self.stepNum = 0
        self.recordGap = 2

        init_obs = self._get_obs()
        self.novelty_window_size = 10
        self.traj_buffer = []  # [init_obs] * 5

        self.novel_autoencoders = []

        self.novel_visitations = []

        self.sum_of_old = 0

        # autoencoder_dir = "../novelty_data/local/autoencoders/"
        # autoencoder1 = load_model(autoencoder_dir + "new_path_finding_biased_autoencoder_1_test.h5")
        # autoencoder2 = load_model(autoencoder_dir + "new_path_finding_biased_autoencoder_2.h5")
        # autoencoder3 = load_model(autoencoder_dir + "new_path_finding_biased_autoencoder_3.h5")
        # autoencoder4 = load_model(autoencoder_dir + "new_path_finding_autoencoder_4.h5")

        # self.novel_autoencoders.append(autoencoder1)
        # self.novel_autoencoders.append(autoencoder2)
        # self.novel_autoencoders.append(autoencoder3)
        # self.novel_autoencoders.append(autoencoder4)

        # print("Autoencoder Model names: ")
        # for i in range(len(self.novel_autoencoders)):
        #     print("Model {} name is {}".format(i, self.novel_autoencoders[i].name))

        # 5 works for manually designed visitation grids
        # self.novelty_factor = 5

        self.novelty_factor = 50

        self.novelDiff = 0
        self.path_data = []

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        # action
        # norm = np.linalg.norm(action)
        # if norm == 0:
        #     action = action * 0
        # else:
        #     action = action / norm

        pos_before = self.point_pos

        tau = np.clip(action, -self.action_scale, self.action_scale)

        # tau = action * self.action_scale
        wall_hit = self.do_simulation_first_order(tau, self.frameskip)

        obs = self._get_obs()

        pos_after = self.point_pos

        if (len(self.path_data) == 2):
            self.path_data.pop(0)

        self.path_data.append([pos_after, -1])

        self.stepNum += 1

        novelRwd = self.calc_novelty_from_autoencoder(obs)

        # print(self.novelDiff)
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

        close_to_wall_penalty = 0  # -min(5, close_to_wall_penalty)

        alive_penalty = -1  # - self.stepNum

        # progress_reward = 100 * (pos_after[0] - pos_before[0])
        horizontal_position_rwd = 0  # 100 * (pos_after[0] - self.original_pos[0])
        vertical_assistance_rwd = 0  # 10 * (pos_after[1] - self.original_pos[1])

        reward = alive_penalty
        # reward = novelRwd
        # novelRwd = (novelRwd) ** 2
        done = False

        if 6 <= self.grid_map[i, j] <= 9:
            reward += 20 * (self.grid_map[i, j] - 5) ** 2
            self.grid_map[i, j] = 0

        if 2 <= self.grid_map[i, j] <= 5:
            done = True
            # print("Sum of accumulated old penalty: ", self.sum_of_old)
            reward += 500 * (self.grid_map[i, j] - 1) ** 2

        # if wall_hit:
        #     done = True
        # if self.sum_of_old > 10:
        #     reward -= 1000
        # done = True

        # + (alive_penalty + close_to_wall_penalty + horizontal_position_rwd + vertical_assistance_rwd)

        return obs, (reward, novelRwd), done, {'Alive penalty': alive_penalty,
                                               'tau': tau, 'Novelty': novelRwd,
                                               'Total Reward': reward}

    def _get_obs(self):
        return np.concatenate([[self.point_pos[0], self.point_pos[1]], [self.point_vel[0], self.point_vel[1]]]).ravel()

    def do_simulation(self, tau, frameskip):

        wall_hit = 0
        for _ in range(frameskip):
            self.point_acc_force = tau
            i_before, j_before = self.pos_to_grid_idx(self.point_pos)
            next_pos = self.point_pos + self.dt * self.point_vel
            i_after, j_after = self.pos_to_grid_idx(next_pos)

            length = len(self.grid_map) - 1
            if i_after < 0 or i_after > length or j_after < 0 or j_after > length or i_before < 0 or i_before > length or j_before < 0 or j_before > length:

                # print(i_before, j_before)
                # print(i_after, j_after)
                wall_hit += 1
                break
            else:
                if self.grid_map[i_after, j_after] == 1:
                    change_dir = np.array([abs(j_after - j_before), abs(i_after - i_before)])

                    vel = np.zeros(len(self.point_vel))
                    if (change_dir[0] == 1):
                        self.point_vel[0] *= -.01
                    if (change_dir[1] == 1):
                        self.point_vel[1] *= -.01
                    wall_hit += 1
                    break

            self.point_pos = self.point_pos + self.dt * self.point_vel
            self.point_vel = self.point_vel + self.dt * (self.point_acc_force / self.point_mass)
        return wall_hit

    def do_simulation_first_order(self, vel, frameskip):
        self.point_vel = vel
        wall_hit = 0

        for _ in range(frameskip):

            i_before, j_before = self.pos_to_grid_idx(self.point_pos)
            next_pos = self.point_pos + self.dt * self.point_vel
            i_after, j_after = self.pos_to_grid_idx(next_pos)

            length = len(self.grid_map) - 1
            if i_after < 0 or i_after > length or j_after < 0 or j_after > length or i_before < 0 or i_before > length or j_before < 0 or j_before > length:

                # print(i_before, j_before)
                # print(i_after, j_after)
                wall_hit += 1
                break
            else:
                if self.grid_map[i_after, j_after] == 1:
                    if wall_hit == 0:
                        change_dir = np.array([abs(j_after - j_before), abs(i_after - i_before)])

                        vel = np.zeros(len(self.point_vel))
                        if (change_dir[0] == 1):
                            self.point_vel[0] *= -.01
                        if (change_dir[1] == 1):
                            self.point_vel[1] *= -.01

                        wall_hit += 1
                    # break
            self.point_pos = self.point_pos + self.dt * self.point_vel
        return wall_hit

    def _reset(self):
        self.point_pos = self.init_pos + self.np_random.uniform(low=-0.01,
                                                                high=0.01,
                                                                size=(2))

        self.point_vel = -np.zeros(2) + self.np_random.uniform(low=-0.01, high=0.01, size=(2))
        self.point_acc_force = -np.zeros(2)
        self.stepNum = 0
        self.sum_of_old = 0
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

                    elif (2 <= self.grid_map[i, j] <= 5):

                        cell.set_color((self.grid_map[i, j] - 1) / 4.0, 0, 0)

                    elif (6 <= self.grid_map[i, j] <= 9):

                        cell.set_color(0, ((self.grid_map[i, j] - 5) / 4.0), ((self.grid_map[i, j] - 5) / 4.0))
                    else:
                        color = np.ones(3)
                        for vis in self.novel_visitations:
                            if vis[i, j] > 0:
                                color[0] -= vis[i, j]
                                color[1] -= vis[i, j]
                                color[2] = 0.5

                        cell.set_color(color[0], color[1], color[2])

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
                path_segment.set_color(0, min(1, avg_color * 0.8), 0)

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

            if (len(self.traj_buffer) == self.novelty_window_size):
                novelDiffList = []
                # Reshape to 1 dimension
                traj_seg = np.array([self.traj_buffer])
                traj_seg = self.normalizeTraj(traj_seg, -self.qLim, self.qLim, -self.dqLim, self.dqLim)
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

                # if self.novelDiff < 0.06:
                #     self.novelDiff = 0
                # else:
                # self.novelDiff = 1 - min(self.novelDiff, 1)

            novelRwd = self.novelty_factor * self.novelDiff
        return novelRwd


class SimplerPathFindingReleasing(SimplerPathFinding):
    def __init__(self):

        SimplerPathFinding.__init__(self)
        self.symm_autoencoder = None

    def _step(self, action):

        pos_before = self.point_pos

        tau = action * self.action_scale

        wall_hit = self.do_simulation(tau, self.frameskip)

        obs = self._get_obs()

        pos_after = self.point_pos

        if (len(self.path_data) == 2):
            self.path_data.pop(0)

        self.path_data.append([pos_after, -1])

        self.stepNum += 1

        novelRwd = self.calc_novelty_from_autoencoder(obs)

        if self.stepNum >= 180:
            novelRwd = 0
            self.novelDiff = 0

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

        close_to_wall_penalty = -min(5, close_to_wall_penalty)
        alive_penalty = -self.stepNum
        progress_reward = 300 * (pos_after[0] - pos_before[0])
        horizontal_position_rwd = 300 * (pos_after[0] - self.original_pos[0])

        # novelRwd = 100 * novelRwd

        reward = 0

        done = False

        if self.grid_map[i, j] == 2:
            done = True
            reward += 2000
        if wall_hit:
            reward -= 1000
            done = True

        reward += alive_penalty + progress_reward + close_to_wall_penalty + horizontal_position_rwd + novelRwd

        # reward = reward + novelRwd

        return obs, reward, done, {'Alive penalty': alive_penalty,
                                   'Progress Reward': progress_reward,
                                   'tau': tau, 'Novelty': novelRwd,
                                   'Unscaled Novelty': self.novelDiff, 'Total Reward': reward, 'Close to Wall Penalty'
                                   : -close_to_wall_penalty}
