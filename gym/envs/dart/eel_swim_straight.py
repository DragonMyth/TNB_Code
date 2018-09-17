import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator

from .simple_water_world import BaseFluidEnhancedAllDirSimulator


class DartEelSwimStraighEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 5, [-1.0] * 5])

        self.action_scale = np.pi / 2.0
        self.frame_skip = 5
        dart_env.DartEnv.__init__(self, 'simple_eel.skel', self.frame_skip, 15, control_bounds, dt=0.002,
                                  disableViewer=True,
                                  custom_world=BaseFluidEnhancedAllDirSimulator)

        utils.EzPickle.__init__(self)

        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

        num_of_dofs = len(self.robot_skeleton.dofs) - len(self.robot_skeleton.joints[0].dofs)

        self.simulation_dt = self.dt * 1.0 / self.frame_skip
        self.Kp = np.diagflat([0.0] * len(self.robot_skeleton.joints[0].dofs) + [4000.0] * num_of_dofs)
        self.Kd = 150 * self.simulation_dt * self.Kp

    def _step(self, a):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        target = np.clip(a, -self.action_scale, self.action_scale)

        target_pos = np.concatenate(([0.0] * 6, target))
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

        angs = np.abs(self.robot_skeleton.q[6::])
        ang_cost_rate = 0.01
        torque_cost_rate = 0.1

        horizontal_rwd = (cur_com[0] - old_com[0]) * 300

        orth_pen = 30 * (np.abs((np.abs(cur_com[1]) - np.abs(old_com[1]))) + np.abs(
            (np.abs(cur_com[2]) - np.abs(old_com[2]))))

        reward = 1 + horizontal_rwd - orth_pen

        notdone = np.isfinite(ob[5::]).all() and (np.abs(angs) < np.pi / 2.0).all()
        done = not notdone
        return ob, reward, done, {'Total Reward': reward, 'Horizontal Reward': horizontal_rwd,
                                  'Orthogonal Penn': -orth_pen}

    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q[4:6], self.robot_skeleton.dq[3:6], self.robot_skeleton.q[6::],
                               self.robot_skeleton.dq[6::]]).ravel()

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
