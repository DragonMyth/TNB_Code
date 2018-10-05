import subprocess
import datetime
import time
import argparse
import multiprocessing
from Util.post_training_process import *

if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='SimplerPathFinding-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--batch_size_per_process',
                        help='Number of samples collected for each process at each iteration', default=1500)
    parser.add_argument('--num_iterations', help='Number of iterations need to be run', default=250)

    parser.add_argument('--data_collect_env', help='Environment used to collect data', default='SimplerPathFinding-v2')
    parser.add_argument('--collect_policy_gap', help='Gap between policies used to collect trajectories', default=5)
    parser.add_argument('--collect_policy_num', help='Number of policies used to collect trajectories', default=10)
    parser.add_argument('--collect_policy_start', help='First policy used to collect trajectories', default=200)
    parser.add_argument('--collect_num_of_trajs', help='Number of trajectories collected per process per policy',
                        default=40)
    parser.add_argument('--ignore_obs', help='Number of Dimensions in the obs that are ignored', default=0)

    args = parser.parse_args()

    env_name = args.env

    # seed = args.seed
    for s in range(7):
        seed = s * 13 + 7 * (s ** 2)
        for i in range(5):
            curr_run = str(i)

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
            data_saving_path = 'data/ppo_' + env_name + '_seed_' + str(seed) + '_run_' + str(curr_run) + '/' + str(st)

            train_policy = subprocess.call(
                'mpirun -np ' + str(
                    cpu_count) + ' python ./running_regimes/two_objs_policy_train.py'
                + ' --env ' + args.env
                + ' --seed ' + str(seed)
                + ' --curr_run ' + curr_run
                + ' --data_saving_path ' + data_saving_path
                + ' --batch_size_per_process ' + str(args.batch_size_per_process)
                + ' --num_iterations ' + str(args.num_iterations)
                , shell=True)

            collect_data = subprocess.call('mpirun -np ' + str(cpu_count) + ' python ./Util/post_training_process.py'
                                           + ' --seed ' + str(seed)
                                           + ' --curr_run ' + str(curr_run)
                                           + ' --data_collect_env ' + str(args.data_collect_env)
                                           + ' --collect_policy_gap ' + str(args.collect_policy_gap)
                                           + ' --collect_policy_num ' + str(args.collect_policy_num)
                                           + ' --collect_policy_start ' + str(args.collect_policy_start)
                                           + ' --collect_num_of_trajs ' + str(args.collect_num_of_trajs)
                                           + ' --policy_saving_path ' + str(data_saving_path)
                                           + ' --ignore_obs ' + str(args.ignore_obs)
                                           , shell=True)

            collected_data_filename = 'novelty_data/local/sampled_paths/' + args.data_collect_env + '_seed_' + str(
                seed) + '_run_' + str(
                curr_run) + '.pkl'

            plot_save_dir = 'novelty_data/local/sampled_paths/plots/' + args.data_collect_env + '_seed_' + str(
                seed) + '_run_' + str(
                curr_run) + '_visited_plot.png'

            plot_path_data(collected_data_filename, plot_save_dir)

            train_autoencoder = subprocess.call('python ./Util/train_traj_autoencoder.py'
                                                + ' --env ' + str(args.env)
                                                + ' --seed ' + str(seed)
                                                + ' --data_collect_env ' + str(args.data_collect_env)
                                                + ' --curr_run ' + str(curr_run)
                                                , shell=True

                                                )
