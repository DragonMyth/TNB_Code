from gym.envs.dart.dart_env import DartEnv
# ^^^^^ so that user gets the correct error
# message if pydart is not installed correctly

from gym.envs.dart.cart_pole import DartCartPoleEnv
from gym.envs.dart.hopper import DartHopperEnv
from gym.envs.dart.cartpole_swingup import DartCartPoleSwingUpEnv
from gym.envs.dart.reacher import DartReacherEnv
from gym.envs.dart.cart_pole_img import DartCartPoleImgEnv
from gym.envs.dart.walker2d import DartWalker2dEnv
from gym.envs.dart.walker3d import DartWalker3dEnv
from gym.envs.dart.inverted_double_pendulum import DartDoubleInvertedPendulumEnv
from gym.envs.dart.dog import DartDogEnv
from gym.envs.dart.reacher2d import DartReacher2dEnv

# Swiming Simulation envs
from gym.envs.dart.eel_swim_straight import DartEelSwimStraighEnv
from gym.envs.dart.eel_swim_straight_data_collect import DartEelSwimStraighDataCollectEnv

# from gym.envs.dart.turtle_swim_straight import DartTurtleSwimStraighEnv
# from gym.envs.dart.turtle_swim_straight_half import DartTurtleSwimStraighHalfEnv
# from gym.envs.dart.turtle_swim_straight_large import DartTurtleSwimStraighLargeEnv
# from gym.envs.dart.turtle_swim_straight_pd import DartTurtleSwimStraighSPDEnv
# from gym.envs.dart.turtle_swim_straight_pd import DartTurtleSwimStraighSPDEnvNoEnf

from gym.envs.dart.turtle_swim_straight_spd import DartTurtleSwimStraighSPDEnv
from gym.envs.dart.turtle_swim_straight_spd_data_collect import DartTurtleSwimStraighSPDEnvDataCollect

from gym.envs.dart.flatworm_swim_straight import DartFlatwormSwimStraightEnv
from gym.envs.dart.flatworm_swim_straight_reduced import DartFlatwormSwimStraightReducedEnv
from gym.envs.dart.flatworm_swim_straight_small import DartFlatwormSwimStraightSmallEnv
from gym.envs.dart.flatworm_swim_straight_small import DartFlatwormSwimStraightSmallFreeBackEnv
from gym.envs.dart.humanoid_swim_straight import DartHumanoidSwimStraightEnv
from gym.envs.dart.humanoid_swim_straight_tau_act import DartHumanoidSwimStraightTauActEnv
from gym.envs.dart.humanoid_swim_straight_legs import DartHumanoidSwimStraightLegsEnv
from gym.envs.dart.humanoid_swim_straight_arms import DartHumanoidSwimStraightArmsEnv
from gym.envs.dart.humanoid_swim_straight_arms import DartHumanoidSwimStraightArmsLockWristEnv
from gym.envs.dart.humanoid_swim_straight import DartHumanoidVisualizationEnv
from gym.envs.dart.humanoid_swim_straight import DartHumanoidStableWaistEnv
from gym.envs.dart.humanoid_swim_straight_upper import DartHumanoidSwimStraightUpperEnforceEnv
from gym.envs.dart.humanoid_swim_straight_upper import DartHumanoidSwimStraightUpperReleaseEnv
from gym.envs.dart.humanoid_swim_straight_full import DartHumanoidSwimStraightFullEnforceEnv
from gym.envs.dart.humanoid_swim_straight_full import DartHumanoidSwimStraightFullReleaseEnv
from gym.envs.dart.humanoid_swim_straight_full_data_collection import DartHumanoidSwimStraightFullDataCollectionEnv

from gym.envs.dart.cube_path_finding import DartCubePathFindingEnforceEnv
from gym.envs.dart.cube_path_finding import DartCubePathFindingReleaseEnv
from gym.envs.dart.cube_path_finding_data_collect import DartCubePathFindingDataCollectEnv
