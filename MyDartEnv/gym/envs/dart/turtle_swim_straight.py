import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator

class DartTurtleSwimStraighEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 8, [-1.0] * 8])
        self.action_scale = 0.002
        dart_env.DartEnv.__init__(self, 'simple_turtle.skel',2 ,16, control_bounds, dt=0.02, disableViewer=True,
                                  custom_world=BaseFluidSimulator)
        utils.EzPickle.__init__(self)
        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C
        self.original_q = self.robot_skeleton.q

    def _step(self, a):
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
        tau[len(self.robot_skeleton.joints[0].dofs)::] = a[:] * self.action_scale
        old_com = self.robot_skeleton.C
        old_q = self.robot_skeleton.q
        self.do_simulation(tau, self.frame_skip)
        cur_com = self.robot_skeleton.C
        cur_q = self.robot_skeleton.q
        ob = self._get_obs()

        angs = np.abs(self.robot_skeleton.q[6::])
        ang_cost_rate = 0.0001
        torque_cost_rate = 0.001
        force_cost = np.sum(np.abs(tau)) * torque_cost_rate
        horizontal_rwd = (cur_com[0] - old_com[0])*500
        orth_pen = 50*(np.abs(cur_com[1]-self.original_com[1]) + np.abs(cur_com[2]-self.original_com[2]))
        rotate_pen = 20*np.sum(np.abs(cur_q[:3]-self.original_q[:3]))
        reward = 1+horizontal_rwd-rotate_pen-orth_pen
        # print('H',horizontal_rwd)
        # print('O',orth_pen)
        # print('R',reward)
        # print(tau)
        # print(tau[len(self.robot_skeleton.joints[0].dofs)::])
        # print(np.abs(cur_com[0])<3)
        # reward = reward-0.001 if not (np.abs(angs)<np.pi/2).all() else reward


        notdone = np.isfinite(ob).all() and (np.abs(angs)<np.pi/2.0).all()
        done = not notdone
        #done = False
        # print(notdone)
        # print(done)
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.q[6::], self.robot_skeleton.dq[6::]]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qpos[:6]=0
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        qvel[:6]=0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(-60)
        self.track_skeleton_id = 0
