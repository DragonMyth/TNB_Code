import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from gym.envs.dart import turtle_swim_straight


class DartTurtleSwimStraighHalfEnv(turtle_swim_straight.DartTurtleSwimStraighEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 4, [-1.0] * 4])

        self.action_scale = 0.004
        dart_env.DartEnv.__init__(self, 'large_flipper_turtle.skel', 2, 16, control_bounds, dt=0.02, disableViewer=False,
                                  custom_world=BaseFluidSimulator)
        utils.EzPickle.__init__(self)
        self.init_state = self._get_obs()

    def _step(self, a):
        a = np.concatenate((a, a))
        return super()._step(a)
