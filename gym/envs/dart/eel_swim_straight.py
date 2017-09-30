import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator

class DartEelSwimStraighEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 5, [-1.0] * 5])
        self.action_scale = 0.0005
        dart_env.DartEnv.__init__(self, 'simple_eel.skel', 2, 10, control_bounds, dt=0.02, disableViewer=False,
                                  custom_world=BaseFluidSimulator)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[0:len(self.robot_skeleton.joints[0].dofs)] = 0
        tau[len(self.robot_skeleton.joints[0].dofs)::] = a[:] * self.action_scale
        old_com = self.robot_skeleton.C
        self.do_simulation(tau, self.frame_skip)
        cur_com = self.robot_skeleton.C
        ob = self._get_obs()

        angs = np.abs(self.robot_skeleton.q[6::])
        ang_cost_rate = 0.001
        torque_cost_rate = 0.01

        horizontal_rwd = (cur_com[0] - old_com[0])*300
        orth_pen = 30*(np.abs((np.abs(cur_com[1]) - np.abs(old_com[1]))) + np.abs((np.abs(cur_com[2]) - np.abs(old_com[2]))))
        reward = horizontal_rwd-orth_pen-ang_cost_rate*np.sum(np.abs(angs))-np.sum(np.abs(tau))*torque_cost_rate
        # print('H',horizontal_rwd)
        # print('O',orth_pen)
        # print('R',reward)
        # print(tau)
        # print(tau[len(self.robot_skeleton.joints[0].dofs)::])
        # print(np.abs(cur_com[0])<3)
        reward = reward-0.001 if not (np.abs(angs)<np.pi/2).all() else reward
        notdone = np.isfinite(ob).all() and (np.abs(cur_com[0]) <= 3) and (np.abs(angs)<6*np.pi/11).all()
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
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.01, high=.01, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -3.5
        self._get_viewer().scene.tb._set_theta(0)
        self.track_skeleton_id = 0
