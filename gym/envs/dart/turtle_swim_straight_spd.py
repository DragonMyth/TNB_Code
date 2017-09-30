import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator


class DartTurtleSwimStraighSPDEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 8, [-1.0] * 8])
        self.action_scale = np.pi / 2.0
        frame_skip = 5
        dart_env.DartEnv.__init__(self, 'large_flipper_turtle.skel', frame_skip, 16, control_bounds, dt=0.002,
                                  disableViewer=False,
                                  custom_world=BaseFluidSimulator)
        utils.EzPickle.__init__(self)
        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

        num_of_dofs = len(self.robot_skeleton.dofs) - len(self.robot_skeleton.joints[0].dofs)

        self.simulation_dt = self.dt * 1.0 / frame_skip
        self.Kp = np.diagflat([0.0] * len(self.robot_skeleton.joints[0].dofs) + [4000.0] * num_of_dofs)
        # self.Kd = 150 * self.simulation_dt * self.Kp
        self.Kd = self.simulation_dt * self.Kp

        self.invM = np.linalg.inv(self.robot_skeleton.M + self.Kd * self.simulation_dt)

    def _step(self, a):
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        old_dq = self.robot_skeleton.dq

        target_pos = np.concatenate(([0.0] * 6, a * self.action_scale))
        ##SPD Controller
        # for i in range(self.frame_skip):
        #     invM = self.invM
        #     p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
        #     d = -self.Kd.dot(self.robot_skeleton.dq)
        #     qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        #     tau = p + d - self.Kd.dot(qddot) * self.simulation_dt
        #     # tau *= 0.0005
        #     tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
        #     self.do_simulation(tau, 1)

        invM = self.invM
        p = -self.Kp.dot(self.robot_skeleton.q + self.robot_skeleton.dq * self.simulation_dt - target_pos)
        d = -self.Kd.dot(self.robot_skeleton.dq)
        qddot = invM.dot(-self.robot_skeleton.c + p + d + self.robot_skeleton.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.simulation_dt
        tau *= 0.0005
        tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
        self.do_simulation(tau,self.frame_skip)
        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        cur_dq = self.robot_skeleton.dq
        ob = self._get_obs()

        angs = np.abs(self.robot_skeleton.q[6::])
        ang_cost_rate = 0.0001
        torque_cost_rate = 0.001
        symm_rate = 0
        symm_pos_pen = symm_rate * np.sum(np.abs(cur_q[
                                                     'left_front_limb_joint_1', 'left_front_limb_joint_2', 'left_rear_limb_joint_1', 'left_rear_limb_joint_2'] -
                                                 cur_q[
                                                     'right_front_limb_joint_1', 'right_front_limb_joint_2', 'right_rear_limb_joint_1', 'right_rear_limb_joint_2']))
        # symm_vel_pen = symm_rate*np.sum(np.abs(cur_dq['left_front_limb_joint_1', 'left_front_limb_joint_2'] - cur_dq[
        #     'right_front_limb_joint_1', 'right_front_limb_joint_2']))
        horizontal_pos_rwd = (cur_com[0] - old_com[0]) * 500
        horizontal_vel_rwd = 3*cur_dq[3]
        orth_pen = (np.abs(cur_com[1] - self.original_com[1]) + np.abs(cur_com[2] - self.original_com[2]))
        rotate_pen = np.sum(np.abs(cur_q[:3] - self.original_q[:3]))
        # mirror_enforce
        reward = 1 + horizontal_pos_rwd +  horizontal_vel_rwd - rotate_pen - orth_pen - symm_pos_pen

        notdone = np.isfinite(ob).all() and (np.abs(angs) < np.pi / 2.0).all()
        done = not notdone

        return ob, reward, done, {'rwd': reward, 'horizontal_pos_rwd': horizontal_pos_rwd,
                                  'horizontal_vel_rwd': horizontal_vel_rwd,
                                  'rotate_pen': -rotate_pen, 'orth_pen': -orth_pen, 'symm_pos_pen': -symm_pos_pen}

    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q[6::], self.robot_skeleton.dq[6::]]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qpos[:6] = 0
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel[:6] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(-60)
        self.track_skeleton_id = 0
