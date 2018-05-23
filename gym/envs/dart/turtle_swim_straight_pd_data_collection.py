import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidEnhancedAllDirSimulator


class DartTurtleSwimStraighPDCollectionEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 8, [-1.0] * 8])
        self.action_scale = np.pi / 2.0
        self.frame_skip = 1
        dart_env.DartEnv.__init__(self, 'large_flipper_turtle_real.skel', self.frame_skip, 21, control_bounds, dt=0.002,
                                  disableViewer=True,
                                  custom_world=BaseFluidEnhancedAllDirSimulator)

        utils.EzPickle.__init__(self)

        num_of_dofs = len(self.robot_skeleton.dofs) - len(self.robot_skeleton.joints[0].dofs)
        self.simulation_dt = self.dt * 1.0 / self.frame_skip

        self.Kp = np.diagflat(
            [0.0] * len(self.robot_skeleton.joints[0].dofs) + [8] * 1 + [4] * 1 + [.1] * 2 + [8] * 1 + [4] * 1 + [
                .1] * 2)
        self.Kd = np.diagflat(
            [0.0] * len(self.robot_skeleton.joints[0].dofs) + [.8] * 1 + [.4] * 1 + [.05] * 2 + [.8] * 1 + [.4] * 1 + [
                .05] * 2)

        # print("Model name is", self.symm_autoencoder.name)

    def _step(self, a):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        target_pos = np.concatenate(([0.0] * 6, a * self.action_scale))

        for _ in range(self.frame_skip):
            p = -self.Kp.dot(self.robot_skeleton.q - target_pos)
            d = -self.Kd.dot(self.robot_skeleton.dq)
            tau = p + d

            tau[0:6] = 0

            self.do_simulation(tau, 1)

        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        # this should be the unscaled novelty term
        novelDiff = 0
        angs = np.abs(self.robot_skeleton.q[6::])
        ang_cost_rate = 0.0001
        torque_cost_rate = 0.001

        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 5000
        rotate_pen = np.sum(np.abs(cur_q[0]))

        energy_consumed_pen = np.sum(np.abs(tau[6::] * old_dq[6::] * self.simulation_dt))

        # mirror_enforce
        reward = 1 + horizontal_pos_rwd - rotate_pen   - energy_consumed_pen

        valid = np.isfinite(ob[5::]).all() and (np.abs(angs) < np.pi / 2.0).all() and (np.abs(cur_dq) < 50).all()
        done = not valid

        return ob, reward, done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                  'rotate_pen': -rotate_pen, 'tau': tau,
                                  'symm_pos_pens': -0, 'novelty': novelDiff,
                                  'energy_consumed_pen': -energy_consumed_pen}

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

    def normalizeTraj(self, traj, minq, maxq, mindq, maxdq):
        traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (maxq - minq)
        traj[:, :, int(len(traj[0, 0]) / 2)::] /= (maxdq - mindq)
        return traj