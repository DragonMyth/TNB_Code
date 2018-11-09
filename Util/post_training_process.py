import random

from matplotlib import cm
from mpi4py import MPI

import gym
import joblib
from joblib import Parallel, delayed
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames

from tkinter.filedialog import askdirectory
import numpy as np

import matplotlib.pyplot as plot
from boltons import iterutils
import multiprocessing
import matplotlib.colors
import os.path as osp

from baselines import bench as bc
from baselines import logger

import tensorflow as tf
from baselines.ppo1 import mlp_policy, mlp_policy_novelty, mlp_policy_mirror_novelty
import baselines.common.tf_util as U

import itertools
from keras.models import load_model
from gym import wrappers


def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
    return mlp_policy_novelty.MlpPolicyNovelty(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64,
                                               num_hid_layers=3,
                                               )


def mirror_turtle_policy_fn(name, ob_space, ac_space):
    return mlp_policy_mirror_novelty.MlpPolicyMirrorNovelty(name=name, ob_space=ob_space, ac_space=ac_space,
                                                            hid_size=64,
                                                            num_hid_layers=3,
                                                            mirror_loss=True,
                                                            observation_permutation=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                     10,
                                                                                     15, 16, 17, 18,
                                                                                     11, 12, 13, 14,
                                                                                     23, 24, 25, 26,
                                                                                     19, 20, 21, 22],

                                                            action_permutation=[4, 5, 6, 7, 0, 1, 2, 3]
                                                            )


def perform_rollout(policy,
                    env,
                    snapshot_dir=None,
                    animate=False,
                    plot_result=False,
                    control_step_skip=1,
                    costum_horizon=None,
                    stochastic=True,
                    debug=False,
                    saved_rollout_path=None,
                    num_repeat=1,
                    record=False

                    ):
    # all_reward_data = dict(rewards=[], horizontal_pos_rwds=[], horizontal_vel_rwds=[], rotate_pens=[], orth_pens=[],
    #                        energy_consumed_pen=[], tau=[], symm_pos_pens=[], novelty=[])

    predefined_actions = None
    if (saved_rollout_path):
        predefined_actions = joblib.load(saved_rollout_path)['actions']
    # print(predefined_actions)
    all_reward_info = []

    for _ in range(num_repeat):

        path = {'observations': [], 'actions': []}
        last_action = None

        horizon = env._max_episode_steps
        if costum_horizon != None:
            horizon = costum_horizon
        if animate:
            # env.env.env.
            if hasattr(env.env, 'disableViewer'):
                env.env.disableViewer = False
            if record:
                env = wrappers.Monitor(env, snapshot_dir + '/../policy_runs', force=True)
        observation = env.reset()

        for i in range(horizon):
            if i % 200 == 0 and debug:
                print("Current Timestep:", i)
            if animate:
                env.render()
            if policy is None:
                action_taken = (np.random.rand(env.unwrapped.action_dim) - 0.5 * np.ones(
                    env.unwrapped.action_dim)) * np.pi
            else:
                if i % control_step_skip == 0:

                    # observation_permutation = [0, 1, 2, 3, 4, 9, 10, 11, 12,
                    #                            5, 6, 7, 8, 17, 18, 19, 20, 13,
                    #                            14, 15, 16]
                    #
                    # action_permutation = [4, 5, 6, 7, 0, 1, 2, 3]
                    # obs_perm_mat = np.zeros((len(observation_permutation), len(observation_permutation)),
                    #                         dtype=np.float32)
                    # act_perm_mat = np.zeros((len(action_permutation), len(action_permutation)), dtype=np.float32)
                    #
                    # for k, perm in enumerate(observation_permutation):
                    #     obs_perm_mat[k][perm] = 1
                    #
                    # for k, perm in enumerate(action_permutation):
                    #     act_perm_mat[k][perm] = 1
                    #
                    # mirr_obs = np.dot(observation, obs_perm_mat)
                    #
                    # orig_mean = policy._orig_mean(observation)
                    # mirr_mean = policy._mirr_mean(observation)
                    # mirr_obs_mean = policy._mirr_obs_mean(observation)
                    # means_1 = (orig_mean, mirr_mean,mirr_obs_mean)
                    # orig_mean_2 = policy._orig_mean(mirr_obs)
                    # mirr_mean_2 = policy._mirr_mean(mirr_obs)
                    # mirr_obs_mean_2 = policy._mirr_obs_mean(mirr_obs)
                    #
                    # means_2 = (orig_mean_2, mirr_mean_2,mirr_obs_mean_2)
                    # action_taken_1 = policy.act(False, observation)[0]
                    # action_taken_2 = policy.act(False, mirr_obs)[0]

                    action_taken = policy.act(stochastic, observation)[0]
                    last_action = action_taken
                else:
                    action_taken = last_action
            if animate:
                if (policy is not None):
                    # log_std = policy.log_std()

                    std = policy.log_std(observation)[0]

                    # print("std is: ", std)
                    # print("Action taken is: ", action_taken)
            if predefined_actions:
                action_taken = predefined_actions[i][::-1]

            observation, reward, done, info = env.step(action_taken)
            # observation, reward, done, info = env.step(np.array([-1, 0]))

            reward = reward[0] if type(reward) == tuple else reward
            path['observations'].append(observation)
            path['actions'].append(action_taken)
            all_reward_info.append(info)

            if done:
                # print("Rollout is Done")
                break

    if plot_result:
        iters = np.arange(1, len(all_reward_info) + 1, 1)
        data_list = {}
        for i in range(len(all_reward_info)):

            for key in all_reward_info[0]:
                if key not in data_list:
                    data_list[key] = []
                data_list[key].append(all_reward_info[i][key])

        random.seed = 41
        cnt = 0

        plot.figure()

        total_ret = np.sum(data_list['rwd'])
        print("Total return of the trajectory is: ", total_ret)

        # # sns.set_palette('hls', len(data_list))
        # for key in sorted(data_list.keys()):
        #     # print(key)
        #     if (key != 'tau'):
        #         cnt += 1
        #         plot.plot(iters, data_list[key],
        #                   label=str(key))
        #         plot.yscale('symlog')
        #
        # plot.xlabel('Time Steps')
        # plot.ylabel('Step Reward')
        # plot.legend()
        # plot.savefig(snapshot_dir + '/rewardDecomposition.jpg')
        #
        # plot.figure()

        # sns.set_palette('hls', len(data_list))
        for key in sorted(data_list.keys()):
            # print(key)
            if (key != 'actions' and key != 'states'):
                cnt += 1
                plot.plot(iters, data_list[key],
                          label=str(key))
                plot.yscale('symlog')

        plot.xlabel('Time Steps')
        plot.ylabel('Step Reward')
        plot.legend()
        plot.savefig(snapshot_dir + '/rewardDecomposition.jpg')

        plot.figure()

        actionArr = np.array(data_list['actions'])
        plot.title('Torque changes along time')
        norm = matplotlib.colors.SymLogNorm(np.min(abs(actionArr)), linscale=np.min(abs(actionArr)))
        plot.imshow(actionArr.transpose(), norm=norm, aspect='auto')
        plot.gray()
        plot.colorbar()

        plot.xlabel('Time Steps')
        plot.ylabel('Dofs')
        plot.legend()
        plot.savefig(snapshot_dir + '/torqueDecomposition.jpg')

        plot.show()

        plot.figure()
        plot.title('State Vec changes along time')
        statesArr = np.array(data_list['states'])

        # norm = matplotlib.colors.SymLogNorm(np.min(abs(statesArr)), linscale=np.min(abs(statesArr)))
        norm = matplotlib.colors.Normalize(-np.pi, np.pi)

        plot.imshow(statesArr.transpose(), norm=norm, aspect='auto')
        plot.gray()
        plot.colorbar()

        # plot.grid(True)

        plot.xlabel('Time Steps')
        plot.ylabel('Dofs')
        plot.legend()
        plot.savefig(snapshot_dir + '/stateDecomposition.jpg')

        plot.show()

    return path


def collect_rollout(policy, environment, rollout_num, ignoreObs, instancesNum=15, obs_skip=15, **opt):
    # For Each rollout
    rank = MPI.COMM_WORLD.Get_rank()
    np.random.seed(42 + rank)
    environment.seed(42 + rank)
    rollout_num_ranges = range(rollout_num)
    num_cores = multiprocessing.cpu_count()

    # print("Rollout number is ", i)
    def saveSingleRollout(i):

        subsampled_paths_per_thread = []
        # print("Rollout number is ", i)
        # obs_skip = 15  # np.random.randint(7, 28)
        num_datapoints = int(2500 / (instancesNum * obs_skip))

        path = perform_rollout(policy, environment, debug=False, animate=opt['animate'], control_step_skip=5)

        useful_path_data = path['observations'][::]
        # useful_path_data_reverse = useful_path_data[::-1]

        # useful_path_data_combined = useful_path_data + useful_path_data_reverse
        split_paths = list(iterutils.chunked_iter(useful_path_data, instancesNum * obs_skip))

        # Split a rollout in to segments of 50 time steps

        if (num_datapoints > len(split_paths) - 1):
            num_datapoints = len(split_paths) - 1
            # print("WHYWHY")

        for j in range(num_datapoints):

            observations = split_paths[j]
            obs_sample = []
            # From each 50 steps subsample instancesNum steps and store

            for k in range(instancesNum):
                obs_sample.append(observations[int(len(observations) / instancesNum) * k][ignoreObs::])
            subsampled_paths_per_thread.append(obs_sample)
            # paths.extend(split_paths)
            # paths.append(path)
        # print(np.shape(subsampled_paths_per_thread))
        return subsampled_paths_per_thread

    results = []
    for i in rollout_num_ranges:
        results.append(saveSingleRollout(i))
    # results = Parallel(n_jobs=num_cores)(delayed(saveSingleRollout)(i) for i in rollout_num_ranges)

    subsampled_paths = itertools.chain(*results)
    return subsampled_paths


def collect_rollouts_from_dir(env_name, num_policies, output_name, ignoreObs, policy_gap=50, start_num=200,
                              traj_per_policy_per_process=100,
                              policy_file_basename='itr_', data_dir='', policy_func=policy_fn, numState=15,
                              obs_skip=15):
    comm = MPI.COMM_WORLD
    directory = data_dir

    env = gym.make(env_name)
    env.seed(MPI.COMM_WORLD.Get_rank() * 1000)
    subsampled_path_per_proc = []
    for i in range(num_policies):
        policy_idx = i * policy_gap + start_num

        policy_filename = policy_file_basename + str(policy_idx)
        policy_filename_full = directory + '/' + policy_filename + '.pkl'

        policy_param = joblib.load(policy_filename_full)

        tf.reset_default_graph()

        with U.make_session(num_cpu=1) as sess:
            pi = policy_func('pi', env.observation_space, env.action_space)

            restore_policy(sess, pi, policy_param)
            subsampled_paths = collect_rollout(pi, env, traj_per_policy_per_process, ignoreObs, animate=False,
                                               instancesNum=numState, obs_skip=obs_skip)

            subsampled_path_per_proc.extend(subsampled_paths)
            # print("Shape of Path for this policy is ", numpyArr.shape)
            # all_subsampled_paths.extend(subsampled_paths)
            # print("Shape of aggregated paths is", np.array(all_subsampled_paths).shape)
            print("Rank: " + str(MPI.COMM_WORLD.Get_rank()) + " Policy num: " + str(
                policy_idx) + ", Paths on this process so far has shape: " + str(
                np.array(subsampled_path_per_proc).shape))

    subsampled_path_per_proc = np.array(subsampled_path_per_proc)
    all_subsampled_paths = comm.gather(subsampled_path_per_proc, root=0)
    # subsampled_paths = None

    if (MPI.COMM_WORLD.Get_rank() == 0):
        import os
        all_subsampled_paths = list(itertools.chain(*all_subsampled_paths))

        # print("The length of the collected array is", len(all_subsampled_paths))
        numpyArr = np.array(all_subsampled_paths)
        print("All paths final shape is ", numpyArr.shape)

        save_dir = os.path.abspath(os.curdir) + "/novelty_data/local/sampled_paths/" + output_name
        joblib.dump(numpyArr, save_dir)
        return save_dir


def render_policy(env, action_skip=1, save_path=False, save_filename="path_finding_policy_rollout_1.pkl", stoch=False,
                  record=False, policy_func=policy_fn, autoencoder_name_list=[]):
    openFileOption = {}
    openFileOption['initialdir'] = '../data/ppo_' + env
    filename = askopenfilename(**openFileOption)

    # DartFlatworm
    # DartHumanSwim
    # environment = "DartTurtle-v3"
    print(filename)
    policy_param = joblib.load(filename)

    snapshot_dir = filename[0:filename.rfind('/') + 1]

    # env = gym.wrappers.Monitor(env, snapshot_dir, force=True)
    with U.single_threaded_session() as sess:
        env = gym.make(env)
        autoencoder_list = []
        for autoencoder_name in autoencoder_name_list:
            autoencoder_list.append(load_model(autoencoder_name))

        env.env.novel_autoencoders = autoencoder_list

        pi = policy_func('pi', env.observation_space, env.action_space)

        restore_policy(sess, pi, policy_param)

        # summary_writer = tf.summary.FileWriter('./tflog', graph=sess.graph)

        path = perform_rollout(pi, env, snapshot_dir=snapshot_dir, animate=True, plot_result=True,
                               stochastic=stoch,
                               control_step_skip=action_skip,
                               saved_rollout_path=None,
                               record=record

                               )
    if save_path:
        joblib.dump(path, save_filename)


def create_visitation_from_dir(env, num_policies, output_name, ignoreObs, policy_gap=50, start_num=200,
                               traj_per_policy_per_process=100,
                               policy_file_basename='itr_', data_dir=''):
    visitation_grid = np.zeros((21, 21), dtype=int)
    directory = data_dir
    env_temp = gym.make(env)

    for i in range(num_policies):
        policy_idx = i * policy_gap + start_num
        print("This is policy " + str(policy_idx))

        policy_filename = policy_file_basename + str(policy_idx)
        policy_filename_full = directory + '/' + policy_filename + '.pkl'

        policy_param = joblib.load(policy_filename_full)

        tf.reset_default_graph()

        with U.make_session(num_cpu=1) as sess:
            pi = policy_fn('pi', env_temp.observation_space, env_temp.action_space)

            restore_policy(sess, pi, policy_param)
            assigning_visitation_grid(pi, env, traj_per_policy_per_process, ignoreObs, visitation_grid, animate=False, )

    if (MPI.COMM_WORLD.Get_rank() == 0):
        # print(visitation_grid)
        visitation_grid = visitation_grid / (visitation_grid.max())
        print(visitation_grid)
        joblib.dump(visitation_grid, "./novelty_data/local/visitation_grid/" + output_name)

        visualize_visitation_grid(visitation_grid)


def assigning_visitation_grid(policy, environment, rollout_num, ignoreObs, visitation_grid, instancesNum=15, **opt):
    # For Each rollout
    rank = MPI.COMM_WORLD.Get_rank()
    np.random.seed(42 + rank)

    rollout_num_ranges = range(rollout_num)
    num_cores = multiprocessing.cpu_count()
    results = []

    def pos_to_grid_idx(pos):
        normalized_pos = (np.round((pos) / (0.25))).astype(int)

        x_idx = normalized_pos[0] + int(len(visitation_grid) / 2)
        y_idx = -normalized_pos[1] + int(len(visitation_grid) / 2)

        return y_idx, x_idx

    for i in range(rollout_num):

        subsampled_paths_per_thread = []
        print("Rollout number is ", i)
        path = perform_rollout(policy, environment, debug=False, animate=opt['animate'], control_step_skip=5)

        obs = path['observations'][200::]
        for ob in obs:
            visited_i, visited_j = pos_to_grid_idx(np.array([ob[0], ob[1]]))
            visitation_grid[visited_i, visited_j] += 1


def visualize_visitation_grid(visitation_grid):
    if visitation_grid is None:
        filename = askopenfilename()
        visitation_grid = joblib.load(filename)

    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig, ax = plt.subplots()

    ax.imshow(visitation_grid, cmap=cm.Greys)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.1)
    ax.set_xticks(np.arange(-0.5, 27, 1))
    ax.set_yticks(np.arange(-0.5, 27, 1))

    plt.show()


def plot_path_data(path_dir=None, save_dir=None):
    if (MPI.COMM_WORLD.Get_rank() == 0):
        print(path_dir)
        autoencoder = None
        if not path_dir:

            openFileOption = {}
            filenames = askopenfilenames(**openFileOption)
        else:
            filenames = [path_dir]

        dataset = []
        for file in filenames:
            dataset.extend(joblib.load(file))
        dataset = np.array(dataset)

        np.random.shuffle(dataset)

        plot.figure()

        max_data = min(5000, len(dataset))
        for i in range(max_data):
            # for data in dataset:
            if (len(dataset[0, :, 0] > 1)):
                plot.plot(dataset[i, :, 0], dataset[i, :, 1])
            else:
                plot.scatter(dataset[i, 0, 0], dataset[i, 0, 1])
        axes = plot.gca()
        axes.set_ylim([-2.5, 2.5])
        axes.set_xlim([-2.5, 2.5])

        if not save_dir:
            plot.show()
        else:
            plot.savefig(save_dir)


def restore_policy(sess, policy, policy_params):
    cur_scope = policy.get_variables()[0].name[0:policy.get_variables()[0].name.find('/')]
    orig_scope = list(policy_params.keys())[0][0:list(policy_params.keys())[0].find('/')]

    for i in range(len(policy.get_variables())):
        assign_op = policy.get_variables()[i].assign(
            policy_params[policy.get_variables()[i].name.replace(cur_scope, orig_scope, 1)])
        sess.run(assign_op)


def create_manual_visitation():
    visitation_grid = np.zeros((27, 27), dtype=int)

    mid_idx = int(len(visitation_grid) / 2)
    # top
    # visitation_grid[1:mid_idx - 1, mid_idx - 1:mid_idx + 2] = 1
    # bot
    visitation_grid[mid_idx + 2:-1, mid_idx - 1:mid_idx + 2] = 1

    # left
    # visitation_grid[mid_idx - 1:mid_idx + 2, 1:mid_idx - 1] = 1
    # right
    # visitation_grid[mid_idx - 1:mid_idx + 2, mid_idx + 2:-1] = 1
    # visitation_grid[-20:-16, 3:-1] = 1

    print(visitation_grid)
    joblib.dump(visitation_grid, "./novelty_data/local/visitation_grid/fill_bot_lane.pkl")

    visualize_visitation_grid(visitation_grid)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--curr_run', help='Current number of runs in the sequence', default=0)
    parser.add_argument('--data_collect_env', help='Environment used to collect data', default='SimplerPathFinding-v2')
    parser.add_argument('--collect_policy_gap', help='Gap between policies used to collect trajectories', default=5)
    parser.add_argument('--collect_policy_num', help='Number of policies used to collect trajectories', default=10)
    parser.add_argument('--collect_policy_start', help='First policy used to collect trajectories', default=100)
    parser.add_argument('--collect_num_of_trajs', help='Number of trajectories collected per process per policy',
                        default=40)
    parser.add_argument('--policy_saving_path', help='Directory for saving the log files for this run')
    parser.add_argument('--ignore_obs', help='Number of Dimensions in the obs that are ignored', default=0)

    parser.add_argument('--policy_fn_type', help='Length of the trajectory segment used for the autoencoder',
                        default='turtle')

    parser.add_argument('--num_states_per_data', help='Number of states to concatenate within a trajectory segment',
                        default=15)
    parser.add_argument('--obs_skip_per_state', help='Number of simulation steps to skip between consecutive states',
                        default=15)

    args = parser.parse_args()

    policy_func = None
    if args.policy_fn_type == 'turtle':
        policy_func = mirror_turtle_policy_fn
    else:
        policy_func = policy_fn

    save_dir = collect_rollouts_from_dir(args.data_collect_env, int(args.collect_policy_num),
                                         args.data_collect_env + '_seed_' + str(
                                             args.seed) + '_run_' + str(args.curr_run) + '.pkl',
                                         int(args.ignore_obs),
                                         policy_gap=int(args.collect_policy_gap),
                                         policy_file_basename='policy_params_',
                                         start_num=int(args.collect_policy_start),
                                         traj_per_policy_per_process=int(args.collect_num_of_trajs),
                                         data_dir=args.policy_saving_path,
                                         policy_func=policy_func,
                                         numState=int(args.num_states_per_data),
                                         obs_skip=int(args.obs_skip_per_state)
                                         )
