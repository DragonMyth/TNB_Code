import random
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
import os.path as osp

from baselines import bench as bc
from baselines import logger

import tensorflow as tf
from baselines.ppo1 import mlp_policy
import baselines.common.tf_util as U

import itertools


def perform_rollout(policy,
                    environment,
                    snapshot_dir=None,
                    animate=False,
                    plot_result=False,
                    control_step_skip=1,
                    costum_horizon=None,
                    stochastic=True,
                    debug=False,
                    saved_rollout_path=None,
                    num_repeat=1
                    ):
    # all_reward_data = dict(rewards=[], horizontal_pos_rwds=[], horizontal_vel_rwds=[], rotate_pens=[], orth_pens=[],
    #                        energy_consumed_pen=[], tau=[], symm_pos_pens=[], novelty=[])

    predefined_actions = None
    if (saved_rollout_path):
        predefined_actions = joblib.load(saved_rollout_path)['actions']
    print(predefined_actions)
    all_reward_info = []

    env = gym.make(environment)

    env = bc.Monitor(env, logger.get_dir() and
                     osp.join(logger.get_dir(), "monitor.json"), allow_early_resets=True)

    # if animate:
    #     try:
    #         env.wrapped_env.env.env.env.unwrapped.disableViewer = False
    #     except(AttributeError):
    #         env.wrapped_env.env.env.unwrapped.disableViewer = False

    for _ in range(num_repeat):
        observation = env.reset()

        path = {'observations': [], 'actions': []}
        last_action = None

        horizon = env.env._max_episode_steps
        if costum_horizon != None:
            horizon = costum_horizon

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
                    action_taken = policy.act(stochastic, observation)[0]
                    last_action = action_taken
                else:
                    action_taken = last_action
            if animate:
                if (policy is not None):
                    # log_std = policy.log_std()

                    std = policy.log_std(observation)[0]

                    print("std is: ", std)
                    print("Action taken is: ", action_taken)
            if predefined_actions:
                action_taken = predefined_actions[i][::-1]
            observation, reward, done, info = env.step(action_taken)
            path['observations'].append(observation)
            path['actions'].append(action_taken)
            all_reward_info.append(info)

            if done:
                print("Visualization is Done")
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

        # sns.set_palette('hls', len(data_list))
        for key in sorted(data_list.keys()):
            print(key)
            if (key != 'tau'):
                cnt += 1
                plot.plot(iters, data_list[key],
                          label=str(key))
                plot.yscale('symlog')

        plot.xlabel('Time Steps')
        plot.ylabel('Step Reward')
        plot.legend()
        plot.savefig(snapshot_dir + '/rewardDecomposition.jpg')

        plot.figure()

        actionArr = np.array(data_list['tau'])
        for i in range(len(actionArr[0])):
            plot.plot(iters, actionArr[:, i], color=(random.random(), random.random(), random.random()),
                      label='Joint_' + str(i))

        plot.xlabel('Time Steps')
        plot.ylabel('Torque Value')
        plot.legend()
        plot.savefig(snapshot_dir + '/torqueDecomposition.jpg')

        plot.show()
    return path


def collect_rollout(policy, environment, rollout_num, ignoreObs, instancesNum=15, **opt):
    # For Each rollout
    rank = MPI.COMM_WORLD.Get_rank()
    np.random.seed(42 + rank)

    np.random.seed(42)

    rollout_num_ranges = range(rollout_num)
    num_cores = multiprocessing.cpu_count()

    # print("Rollout number is ", i)
    def saveSingleRollout(i):

        subsampled_paths_per_thread = []
        print("Rollout number is ", i)
        obs_skip = np.random.randint(7, 29)
        num_datapoints = int(2500 / (instancesNum * 28))

        path = perform_rollout(policy, environment, debug=False, animate=opt['animate'], control_step_skip=5)

        useful_path_data = path['observations'][:]
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
        print(np.shape(subsampled_paths_per_thread))
        return subsampled_paths_per_thread

    results = []
    for i in rollout_num_ranges:
        results.append(saveSingleRollout(i))
    # results = Parallel(n_jobs=num_cores)(delayed(saveSingleRollout)(i) for i in rollout_num_ranges)

    subsampled_paths = itertools.chain(*results)
    return subsampled_paths


def collect_rollouts_from_dir(env, num_policies, output_name, ignoreObs, policy_gap=50, start_num=200,
                              traj_per_policy_per_process=100,
                              policy_file_basename='itr_', ):
    comm = MPI.COMM_WORLD
    # directory = None  # './data/ppo_PathFinding-v0/baseline_seed=0'
    directory = './data/ppo_PathFinding-v0/baseline_seed=0'
    # openFileOption = {}
    # openFileOption['initialdir'] = './data/ppo_' + env
    # directory = askdirectory(**openFileOption)

    all_subsampled_paths = None
    if (MPI.COMM_WORLD.Get_rank() == 0):
        all_subsampled_paths = []

    env_temp = gym.make(env)

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=3,
                                    )

    subsampled_path_per_proc = []
    for i in range(num_policies):
        policy_idx = i * policy_gap + start_num
        print("This is policy " + str(policy_idx))

        policy_filename = policy_file_basename + str(policy_idx)
        policy_filename_full = directory + '/' + policy_filename
        tf.reset_default_graph()

        with U.make_session(num_cpu=1) as sess:
            pi = policy_fn('pi', env_temp.observation_space, env_temp.action_space)

            saver = tf.train.Saver()
            saver.restore(sess, policy_filename_full)

            print(policy_filename_full)

            subsampled_paths = collect_rollout(pi, env, traj_per_policy_per_process, ignoreObs, animate=False)

            subsampled_path_per_proc.extend(subsampled_paths)
            # print("Shape of Path for this policy is ", numpyArr.shape)
            # all_subsampled_paths.extend(subsampled_paths)
            # print("Shape of aggregated paths is", np.array(all_subsampled_paths).shape)

    subsampled_path_per_proc = np.array(subsampled_path_per_proc)
    all_subsampled_paths = comm.gather(subsampled_path_per_proc, root=0)
    # subsampled_paths = None
    if (MPI.COMM_WORLD.Get_rank() == 0):
        all_subsampled_paths = list(itertools.chain(*all_subsampled_paths))

        # print("The length of the collected array is", len(all_subsampled_paths))
        numpyArr = np.array(all_subsampled_paths)
        print("All paths final shape is ", numpyArr.shape)

        joblib.dump(numpyArr, "./novelty_data/local/sampled_paths/" + output_name)


def render_policy(env, action_skip=1, save_path=False, save_filename="path_finding_policy_rollout_1.pkl"):
    openFileOption = {}
    openFileOption['initialdir'] = './data/ppo_' + env
    filename = askopenfilename(**openFileOption)

    # DartFlatworm
    # DartHumanSwim
    # environment = "DartTurtle-v3"

    env_temp = gym.make(env)

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=3,
                                    )

    snapshot_dir = filename[0:filename.rfind('/') + 1]

    # logger.set_snapshot_dir(snapshot_dir)
    with U.single_threaded_session() as sess:
        pi = policy_fn('pi', env_temp.observation_space, env_temp.action_space)

        saver = tf.train.Saver()

        filename = filename.split('.')[:-1][0]

        saver.restore(sess, filename)

        path = perform_rollout(pi, env, snapshot_dir=snapshot_dir, animate=True, plot_result=True,
                               stochastic=False,
                               control_step_skip=action_skip,
                               saved_rollout_path=None
                               )
    if save_path:
        joblib.dump(path, save_filename)


def plot_path_data():
    autoencoder = None

    openFileOption = {}
    filenames = askopenfilenames(**openFileOption)

    dataset = []
    for file in filenames:
        dataset.extend(joblib.load(file))
    dataset = np.array(dataset)

    np.random.shuffle(dataset)

    plot.figure()

    for data in dataset:
        plot.plot(data[:, 0], data[:, 1])

    axes = plot.gca()
    axes.set_ylim([-2.5, 2.5])
    axes.set_xlim([-2.5, 2.5])
    plot.show()


# collect_rollouts(10)
# render_policy('DartHumanUpperSwim-v1')
# render_policy('DartTurtle-v5')

# render_policy('DartTurtle-v7', action_skip=5)
# render_policy('DartHumanFullSwim-v1')
# render_policy('DartHumanFullSwim-v2', action_skip=5)

# render_policy('SimplerPathFinding-v2', action_skip=5)
# render_policy('PathFinding-v2', action_skip=5)
# perform_rollout(None, 'PathFinding-v0', animate=True)

# render_policy('DartPathFinding-v0')
# render_policy('DartPathFinding-v2', action_skip=5)

# perform_rollout(None,'DartHumanFullSwim-v0',animate=True)
# perform_rollout(None,'DartTurtle-v3',animate=True)

# perform_rollout(None,'DartTurtle-v3',animate=True)


# collect_rollout_from_policy('humanoid_sample.pkl')
# collect_rollouts_from_dir(20, 'Asymmetric_progress_data.pkl')
# collect_rollouts_from_dir(20, 'symmetric_progress_data.pkl',policy_file_basename='policy_')

# collect_rollouts_from_dir(20, 'policy_4.pkl')
# collect_rollouts_from_dir(20, 'policy_4.pkl')
# collect_rollouts_from_dir('DartHumanUpperSwim-v1', 20, 'humanoid_from_autoencoder1+2+3+4.pkl', 9)
# collect_rollouts_from_dir('DartHumanFullSwim-v1', 20, 'humanoid_from_autoencoder1+2+3.pkl', 9)

# collect_rollouts_from_dir('DartTurtle-v7', 20, 'turtle_from_autoencoder1+2+3.pkl', 5, start_num=0, traj_per_policy=100)
#
# collect_rollouts_from_dir('DartPathFinding-v2', 20, 'path_finding_from_autoencoder1+2+3.pkl', 0, start_num=0,
#                           traj_per_policy=100)

# collect_rollouts_from_dir('PathFinding-v2', 20, 'path_finding_biased_baseline.pkl', 0, start_num=0,
#                           traj_per_policy=200)

#
# collect_rollouts_from_dir('PathFinding-v2', 4, 'path_finding_biased_baseline.pkl', 0, start_num=50,
#                           traj_per_policy=500)
#
# plot_path_data()

#
# collect_rollouts_from_dir('SimplerPathFinding-v0', 5, 'simpler_path_finding_biased_baseline.pkl', 0, policy_gap=5,
#                           policy_file_basename='policy_param_-',
#                           start_num=0,
#                           traj_per_policy=500)

collect_rollouts_from_dir('SimplerPathFinding-v2', 5, 'simpler_path_finding_biased_baseline.pkl', 0, policy_gap=5,
                          policy_file_basename='policy_param_-',
                          start_num=0,
                          traj_per_policy_per_process=50)
