import subprocess, os
import numpy as np

env = 'DartTurtle-v4'
# env = 'DartHopper-v2'
# env = 'DartReacher3d-v2'  #

norm_scale = np.array([8, np.pi, 8, 25])
# norm_scale = np.array([3, np.pi/3.0, 3, 10])
# norm_scale = np.array([10, 10, 5, np.pi, 5, 5])

norm_scale_str = ''
for i in norm_scale:
    norm_scale_str += str(i) + ' '
my_env = os.environ.copy()
my_env["PATH"] = "/anaconda3/envs/deepRL/bin:" + my_env["PATH"]

train_autoencoder = subprocess.call('python ./Util/train_traj_autoencoder.py'
                                    + ' --env ' + str(env)
                                    + ' --seed ' + str(0)
                                    + ' --data_collect_env ' + str(env)
                                    + ' --curr_run ' + str(0)
                                    + ' --num_epoch ' + str(30)
                                    + ' --batch_size ' + str(1024)
                                    + ' --norm_scales ' + str(norm_scale_str)
                                    , shell=True, env=my_env
                                    )
