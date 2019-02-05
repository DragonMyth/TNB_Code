import argparse
import subprocess
import numpy as np
import time
import datetime

cpu_count = 8  # multiprocessing.cpu_count()
num_sample_per_iter = 12000
num_trajs_per_pol = 200
print("Number of processes: ", cpu_count)
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', default='DartReacher3d-v1')
parser.add_argument('--seed', help='RNG seed', type=int, default=0)
parser.add_argument('--batch_size_per_process',
                    help='Number of samples collected for each process at each iteration',
                    default=int(num_sample_per_iter / cpu_count))
print("Number of samples per process is: ", int(num_sample_per_iter / cpu_count))
parser.add_argument('--num_iterations', help='Number of iterations need to be run', default=400)

parser.add_argument('--data_collect_env', help='Environment used to collect data',
                    default='DartReacher3d-v1')
parser.add_argument('--collect_policy_gap', help='Gap between policies used to collect trajectories', default=5)
parser.add_argument('--collect_policy_num', help='Number of policies used to collect trajectories', default=10)
parser.add_argument('--collect_policy_start', help='First policy used to collect trajectories', default=350)
parser.add_argument('--collect_num_of_trajs', help='Number of trajectories collected per process per policy',
                    default=int(num_trajs_per_pol / cpu_count))

parser.add_argument('--ignore_obs', help='Number of Dimensions in the obs that are ignored', default=6)

parser.add_argument('--num_states_per_data', help='Number of states to concatenate within a trajectory segment',
                    default=10)
parser.add_argument('--obs_skip_per_state', help='Number of simulation steps to skip between consecutive states',
                    default=2)

args = parser.parse_args()
env_name = args.env
seed = args.seed

num_epoch = 200
batch_size = 1024

qnorm = 50
dqnorm = 2 * np.pi
# for s in range(7):
#     seed = s * 13 + 7 * (s ** 2)
norm_scale = np.array([5, np.pi, 5, 50])
norm_scale_str = ''
for i in norm_scale:
    norm_scale_str += str(i) + ' '
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
# i = 0
curr_run = str(0)

data_saving_path = 'data/local/' + str(st) + '_' + env_name + '/ppo_' + env_name + '_seed_' + str(
    seed) + '_run_' + str(curr_run)

train_autoencoder = subprocess.call('OMP_NUM_THREADS="1" python ./Util/train_traj_autoencoder.py'
                                    + ' --env ' + str(args.env)
                                    + ' --seed ' + str(seed)
                                    + ' --data_collect_env ' + str(args.data_collect_env)
                                    + ' --curr_run ' + str(curr_run)
                                    + ' --qnorm ' + str(qnorm)
                                    + ' --dqnorm ' + str(dqnorm)
                                    + ' --num_epoch ' + str(num_epoch)
                                    + ' --batch_size ' + str(batch_size)
                                    + ' --norm_scales ' + str(norm_scale_str)
                                    , shell=True

                                    )
