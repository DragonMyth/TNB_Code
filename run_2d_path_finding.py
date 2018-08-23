#!/usr/bin/env python3
import gym, logging
from gym import wrappers
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
import tensorflow as tf
import multiprocessing


def callback(localv, globalv):
    iters_so_far = localv['iters_so_far']
    local_kwargs = localv['kwargs']

    sess = local_kwargs['session']
    gap = local_kwargs['gap']
    save_dir = logger.get_dir()

    if iters_so_far % gap == 0:
        print("ITER!!")

        saver = tf.train.Saver()
        saver.save(sess, save_dir + '/policy_param_', global_step=iters_so_far)


def train(sess, env_id, num_timesteps, seed):
    from baselines.ppo1 import pposgd_simple, mlp_policy

    rank = MPI.COMM_WORLD.Get_rank()

    workerseed = seed * 50 + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)

    env = gym.make(env_id)

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=3,
                                    )

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)))

    env.seed(seed + rank)

    model = pposgd_simple.learn(env, policy_fn,
                                max_timesteps=num_timesteps,
                                # max_iters=30,
                                timesteps_per_actorbatch=int(2000),
                                clip_param=0.2, entcoeff=0.01,
                                optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
                                gamma=0.99, lam=0.95,
                                schedule='linear',
                                callback=callback,

                                # Following params are kwargs
                                session=sess,
                                gap=5,
                                )

    env.close()
    return model


def main():
    num_iterations = 1
    num_processes = MPI.COMM_WORLD.Get_size()
    num_timesteps_per_process = 2000
    env_name = 'SimplerPathFinding-v0'
    for i in range(5):
        import baselines.common.tf_util as U
        tf.reset_default_graph()
        with U.single_threaded_session() as sess:
            seed = 13 * i + 7 * (i ** 2)

            logger.reset()
            logger.configure('data/ppo_PathFinding-v0/baseline_seed=' + str(seed))

            model = train(sess, env_name,
                          num_timesteps=num_iterations * num_processes * num_timesteps_per_process, seed=seed)

            comm = MPI.COMM_WORLD
            mpi_rank = comm.Get_rank()

            if mpi_rank == 0:
                env = gym.make(env_name)

                # env = wrappers.Monitor(env, logger.get_dir() + '/results', force=True)

                obs = env.reset()

                step = 0
                while (True):

                    env.render()
                    actions = model._act(False, obs)
                    obs, _, done, _ = env.step(actions[0][0])
                    env.render()
                    if done:
                        obs = env.reset()
                    if done:
                        print("Visualization is Done")
                        break

                    step += 1


if __name__ == '__main__':
    main()
