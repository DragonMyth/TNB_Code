import subprocess
import datetime
import time
import argparse
import multiprocessing
from Util.post_training_process import *

if __name__ == '__main__':
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
    parser.add_argument('--num_iterations', help='Number of iterations need to be run', default=500)

    parser.add_argument('--data_collect_env', help='Environment used to collect data',
                        default='DartReacher3d-v1')
    parser.add_argument('--collect_policy_gap', help='Gap between policies used to collect trajectories', default=5)
    parser.add_argument('--collect_policy_num', help='Number of policies used to collect trajectories', default=10)
    parser.add_argument('--collect_policy_start', help='First policy used to collect trajectories', default=450)
    parser.add_argument('--collect_num_of_trajs', help='Number of trajectories collected per process per policy',
                        default=int(num_trajs_per_pol / cpu_count))

    parser.add_argument('--ignore_obs', help='Number of Dimensions in the obs that are ignored', default=6)

    parser.add_argument('--num_states_per_data', help='Number of states to concatenate within a trajectory segment',
                        default=15)
    parser.add_argument('--obs_skip_per_state', help='Number of simulation steps to skip between consecutive states',
                        default=3)
    parser.add_argument('--control_step_skip', help='Number of simulation steps sharing the same control signal',
                        default=1)

    args = parser.parse_args()
    env_name = args.env
    seed = args.seed

    num_epoch = 200
    batch_size = 1024

    # qnorm = 2 * np.pi
    # dqnorm = 50

    norm_scale = np.array([10, 1, 5, 2 * np.pi, 5, 50])
    norm_scale_str = ''
    for i in norm_scale:
        norm_scale_str += str(i) + ' '

    # for s in range(7):
    #     seed = s * 13 + 7 * (s ** 2)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

    specified_time = None
    for i in range(0, 6, 1):

        # i = 0
        curr_run = str(i)
        if i == 0:
            specified_time = '2018-11-07_19:49:55'
        else:
            specified_time = None

        if specified_time is None:
            data_saving_path = 'data/local/' + str(st) + '_' + env_name + '/ppo_' + env_name + '_seed_' + str(
                seed) + '_run_' + str(curr_run)
            train_policy = subprocess.call(
                'OMP_NUM_THREADS="1" mpirun -np ' + str(
                    cpu_count) + ' python ./running_regimes/two_objs_policy_train.py'
                + ' --env ' + args.env
                + ' --seed ' + str(seed)
                + ' --curr_run ' + curr_run
                + ' --data_saving_path ' + str(data_saving_path + '/policy')
                + ' --batch_size_per_process ' + str(args.batch_size_per_process)
                + ' --num_iterations ' + str(args.num_iterations)
                , shell=True)

        else:
            data_saving_path = 'data/local/' + str(
                specified_time) + '_' + env_name + '/ppo_' + env_name + '_seed_' + str(
                seed) + '_run_' + str(curr_run)

        collect_data = subprocess.call(
            'OMP_NUM_THREADS="1" mpirun -np ' + str(cpu_count) + ' python ./Util/post_training_process.py'
            + ' --seed ' + str(seed)
            + ' --curr_run ' + str(curr_run)
            + ' --data_collect_env ' + str(args.data_collect_env)
            + ' --collect_policy_gap ' + str(args.collect_policy_gap)
            + ' --collect_policy_num ' + str(args.collect_policy_num)
            + ' --collect_policy_start ' + str(args.collect_policy_start)
            + ' --collect_num_of_trajs ' + str(args.collect_num_of_trajs)
            + ' --policy_saving_path ' + str(data_saving_path + '/policy')
            + ' --ignore_obs ' + str(args.ignore_obs)
            + ' --policy_fn_type ' + 'normal'
            + ' --num_states_per_data ' + str(args.num_states_per_data)
            + ' --obs_skip_per_state ' + str(args.obs_skip_per_state)
            + ' --control_step_skip ' + str(args.control_step_skip)
            , shell=True)
        #
        collected_data_filename = 'novelty_data/local/sampled_paths/' + args.data_collect_env + '_seed_' + str(
            seed) + '_run_' + str(
            curr_run) + '.pkl'

        plot_save_dir = 'novelty_data/local/sampled_paths/plots/' + args.data_collect_env + '_seed_' + str(
            seed) + '_run_' + str(
            curr_run) + '_visited_plot.png'

        plot_path_data(collected_data_filename, plot_save_dir)

        train_autoencoder = subprocess.call('OMP_NUM_THREADS="1" python ./Util/train_traj_autoencoder.py'
                                            + ' --env ' + str(args.env)
                                            + ' --seed ' + str(seed)
                                            + ' --data_collect_env ' + str(args.data_collect_env)
                                            + ' --curr_run ' + str(curr_run)
                                            # + ' --qnorm ' + str(qnorm)
                                            # + ' --dqnorm ' + str(dqnorm)
                                            + ' --num_epoch ' + str(num_epoch)
                                            + ' --batch_size ' + str(batch_size)
                                            + ' --norm_scales ' + str(norm_scale_str)
                                            , shell=True

                                            )
