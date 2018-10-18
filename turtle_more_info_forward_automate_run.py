import subprocess
import datetime
import time
import argparse
import multiprocessing
from Util.post_training_process import *

if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()
    num_sample_per_iter = 12000
    print("Number of processes: ", cpu_count)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='DartTurtle-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--batch_size_per_process',
                        help='Number of samples collected for each process at each iteration',
                        default=int(num_sample_per_iter / cpu_count))
    print("Number of samples per process is: ", int(num_sample_per_iter / cpu_count))
    parser.add_argument('--num_iterations', help='Number of iterations need to be run', default=600)

    parser.add_argument('--data_collect_env', help='Environment used to collect data', default='DartTurtle-v5')
    parser.add_argument('--collect_policy_gap', help='Gap between policies used to collect trajectories', default=5)
    parser.add_argument('--collect_policy_num', help='Number of policies used to collect trajectories', default=15)
    parser.add_argument('--collect_policy_start', help='First policy used to collect trajectories', default=500)
    parser.add_argument('--collect_num_of_trajs', help='Number of trajectories collected per process per policy',
                        default=20)

    parser.add_argument('--ignore_obs', help='Number of Dimensions in the obs that are ignored', default=11)
    args = parser.parse_args()
    env_name = args.env
    seed = args.seed

    num_epoch = 400
    batch_size = 1024
    qnorm = np.pi
    dqnorm = 50
    # for s in range(7):
    #     seed = s * 13 + 7 * (s ** 2)
    for i in range(6):
        # i = 0
        curr_run = str(i)

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

        # data_saving_path = 'data/ppo_' + env_name + '_seed_' + str(seed) + '_run_' + str(
        #     curr_run) + '/' + '2018-10-01_16:53:21'
        data_saving_path = 'data/ppo_' + env_name + '_seed_' + str(seed) + '_run_' + str(curr_run) + '/' + str(st)

        train_policy = subprocess.call(
            'OMP_NUM_THREADS="1" mpirun -np ' + str(
                cpu_count) + ' python ./running_regimes/turtle_more_info_two_objs_mirror_policy_train.py'
            + ' --env ' + args.env
            + ' --seed ' + str(seed)
            + ' --curr_run ' + curr_run
            + ' --data_saving_path ' + data_saving_path
            + ' --batch_size_per_process ' + str(args.batch_size_per_process)
            + ' --num_iterations ' + str(args.num_iterations)
            , shell=True)

        collect_data = subprocess.call(
            'OMP_NUM_THREADS="1" mpirun -np ' + str(cpu_count) + ' python ./Util/post_training_process.py'
            + ' --seed ' + str(seed)
            + ' --curr_run ' + str(curr_run)
            + ' --data_collect_env ' + str(args.data_collect_env)
            + ' --collect_policy_gap ' + str(args.collect_policy_gap)
            + ' --collect_policy_num ' + str(args.collect_policy_num)
            + ' --collect_policy_start ' + str(args.collect_policy_start)
            + ' --collect_num_of_trajs ' + str(args.collect_num_of_trajs)
            + ' --policy_saving_path ' + str(data_saving_path)
            + ' --ignore_obs ' + str(args.ignore_obs)
            + ' --policy_fn_type ' + 'turtle'
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
                                            + ' --qnorm ' + str(qnorm)
                                            + ' --dqnorm ' + str(dqnorm)
                                            + ' --num_epoch ' + str(num_epoch)
                                            + ' --batch_size ' + str(batch_size)
                                            , shell=True

                                            )
