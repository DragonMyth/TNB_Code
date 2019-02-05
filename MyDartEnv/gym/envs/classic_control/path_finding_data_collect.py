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

logger = logging.getLogger(__name__)


class PathFindingDataCollect(gym.Env):
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
        goal_i, goal_j = 11, 18
        self.grid_map[goal_i, goal_j] = 2

        self.grid_size = 0.25
        self.grid_vis_size = 30

        self.point_mass = 10
        self.point_pos = -2 * np.ones(2)
        self.point_vel = -np.zeros(2)
        self.point_acc_force = -np.zeros(2)

        self.goal_pos = self.grid_idx_to_pos(goal_i, goal_j)

        self.dt = 0.002
        self.frameskip = 1

        self.action_scale = 100
        self.obs_dim = 6

        self.action_space = spaces.Box(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

        obs_high = np.inf * np.ones(self.obs_dim)
        obs_low = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt)) / self.frameskip
        }
        # self.point_mass_trans = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):

        pos_before = self.point_pos

        old_to_goal_dist = (pos_before[0] - self.goal_pos[0]) ** 2 + (pos_before[1] - self.goal_pos[1]) ** 2

        tau = action * self.action_scale

        self.do_simulation(tau, self.frameskip)

        obs = self._get_obs()

        pos_after = self.point_pos

        curr_to_goal_dist = (pos_after[0] - self.goal_pos[0]) ** 2 + (pos_after[1] - self.goal_pos[1]) ** 2
        #
        alive_penalty = -5
        progress_reward = 80 * (old_to_goal_dist - curr_to_goal_dist)
        distance_to_goal_reward = min(70, 10 / curr_to_goal_dist)

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
        if self.grid_map[i, j] == 1:
            done = True
            reward -= 2000
        elif self.grid_map[i, j] == 2:
            done = True
            reward += 10000
        return obs, reward, done, {'Alive penalty': alive_penalty,
                                   'Progress Reward': progress_reward,
                                   'Distance to Goal Reward': distance_to_goal_reward, 'tau': tau,
                                   'Total Reward': reward,
                                   'Close to Wall Penalty'
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
                wall_hit = 1

            self.point_pos = self.point_pos + self.dt * self.point_vel
            self.point_vel = self.point_vel + self.dt * (self.point_acc_force / self.point_mass)
        return wall_hit

    def _reset(self):

        self.point_pos = -2 * np.ones(2) + self.np_random.uniform(low=-0.01, high=0.01, size=(2))

        self.point_vel = -np.zeros(2) + self.np_random.uniform(low=-0.01, high=0.01, size=(2))
        self.point_acc_force = -np.zeros(2)

        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800

        if self.viewer is None:
            from gym.envs.classic_control import rendering
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

                    cell.add_attr(rendering.Transform())

                    if (self.grid_map[i, j] == 1):
                        cell.set_color(0, 0, 0)

                    elif (self.grid_map[i, j] == 2):
                        cell.set_color(1, 0, 0)
                    else:
                        cell.set_color(1, 1, 1)

                    self.viewer.add_geom(cell)

                    right_edge = rendering.Line((l, b), (l, t))

                    bottom_edge = rendering.Line((l, b), (r, b))

                    right_edge.set_color(0.5, 0, 0)
                    bottom_edge.set_color(0.5, 0, 0)
                    self.viewer.add_geom(right_edge)

                    self.viewer.add_geom(bottom_edge)
            #
            q = self.point_pos

            q_x = screen_width / 2

            q_y = screen_height / 2

            point_mass = rendering.FilledPolygon(
                [(q_x - 5, q_y - 5), (q_x - 5, q_y + 5), (q_x + 5, q_y + 5), (q_x + 5, q_y - 5)])
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
