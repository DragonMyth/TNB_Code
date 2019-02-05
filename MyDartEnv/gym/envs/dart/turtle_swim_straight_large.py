import numpy as np
from gym import utils
from gym.envs.dart import dart_env
from .simple_water_world import BaseFluidSimulator
from gym.envs.dart import turtle_swim_straight


class DartTurtleSwimStraighLargeEnv(turtle_swim_straight.DartTurtleSwimStraighEnv, utils.EzPickle):
    def __init__(self):
        control_bounds = np.array([[1.0] * 8, [-1.0] * 8])

        self.action_scale = 0.002
        dart_env.DartEnv.__init__(self, 'large_flipper_turtle.skel', 4, 16, control_bounds, dt=0.002, disableViewer=False,
                                  custom_world=BaseFluidSimulator)
        utils.EzPickle.__init__(self)
        self.init_state = self._get_obs()
        self.original_com = self.robot_skeleton.C