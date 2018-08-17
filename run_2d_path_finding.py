#!/usr/bin/env python3
import gym
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger


def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import pposgd_simple, mlp_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)

    env = gym.make(env_id)

    def policy_fn(name, ob_space, ac_space):  # pylint: disable=W0613
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space, hid_size=64, num_hid_layers=3,
                                    )

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    model = pposgd_simple.learn(env, policy_fn,
                                max_timesteps=1e5,
                                timesteps_per_actorbatch=int(3000),
                                clip_param=0.2, entcoeff=0.01,
                                optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
                                gamma=0.99, lam=0.95,
                                schedule='linear'
                                )

    env.close()
    return model


def main():
    model = train('PathFinding-v0', num_timesteps=500, seed=42)

    env = gym.make('PathFinding-v0')

    obs = env.reset()

    while (True):

        env.render()
        actions = model._act(True, obs)
        obs, _, done, _ = env.step(actions[0][0])
        env.render()
        if done:
            obs = env.reset()
        if done:
            print("Visualization is Done")
            break


if __name__ == '__main__':
    main()
