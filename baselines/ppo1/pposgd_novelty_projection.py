from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque


def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0  # return in current episode
    cur_ep_ret_novel = 0  # Novelty Return in current episode

    cur_ep_len = 0  # len of current episode
    ep_rets = []  # returns of completed episodes in this segment
    ep_rets_novel = []  # novelty returns of completed episodes in this segment
    ep_lens = []  # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    rews_novel = np.zeros(horizon, 'float32')

    vpreds = np.zeros(horizon, 'float32')
    vpreds_novel = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, vpred_novel = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "rew_novel": rews_novel, "vpred": vpreds, "vpred_novel": vpreds_novel,
                   "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "nextvpred_novel": vpreds_novel * (1 - new),
                   "ep_rets": ep_rets, "ep_rets_novel": ep_rets_novel, "ep_lens": ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_rets_novel = []
            ep_lens = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        vpreds_novel[i] = vpred_novel

        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)  # rew is expected to be a tuple with base and novelty reward

        rews[i] = rew[0]
        rews_novel[i] = rew[1]

        cur_ep_ret += rew[0]
        cur_ep_ret_novel += rew[1]
        cur_ep_len += 1

        if new:
            ep_rets.append(cur_ep_ret)
            ep_rets_novel.append(cur_ep_ret_novel)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_ret_novel = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    vpred_novel = np.append(seg["vpred_novel"], seg["nextvpred_novel"])

    T = len(seg["rew"])

    seg["adv"] = gaelam = np.empty(T, 'float32')
    seg["adv_novel"] = gaelam_novel = np.empty(T, 'float32')

    rew = seg["rew"]
    rew_novel = seg["rew_novel"]
    lastgaelam = 0
    lastgaelam_novel = 0

    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

        delta_novel = rew_novel[t] + gamma * vpred_novel[t + 1] * nonterminal - vpred_novel[t]
        gaelam_novel[t] = lastgaelam_novel = delta_novel + gamma * lam * nonterminal * lastgaelam_novel

    seg["tdlamret"] = seg["adv"] + seg["vpred"]
    seg["tdlamret_novel"] = seg["adv_novel"] + seg["vpred_novel"]


def learn(env, policy_fn, *,
          timesteps_per_actorbatch,  # timesteps per actor per update
          clip_param, entcoeff,  # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,  # optimization hypers
          gamma, lam,  # advantage estimation
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None,  # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant',  # annealing for stepsize parameters (epsilon and adam)
          **kwargs,
          ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    pi = policy_fn("pi", ob_space, ac_space)  # Construct network for new policy

    oldpi = policy_fn("oldpi", ob_space, ac_space)  # Network for old policy

    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    atarg_novel = tf.placeholder(dtype=tf.float32,
                                 shape=[None])  # Target advantage function for the novelty reward term
    ret_novel = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return for the novelty reward term

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32,
                            shape=[])  # learning rate multiplier, updated with schedule

    clip_param = clip_param * lrmult  # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()

    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # pnew / pold

    surr1 = ratio * atarg  # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg  #

    surr1_novel = ratio * atarg_novel  # surrogate loss of the novelty term
    surr2_novel = tf.clip_by_value(ratio, 1.0 - clip_param,
                                   1.0 + clip_param) * atarg_novel  # surrogate loss of the novelty term

    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO's pessimistic surrogate (L^CLIP)
    pol_surr_novel = -tf.reduce_mean(tf.minimum(surr1_novel, surr2_novel))  # PPO's surrogate for the novelty part

    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    vf_loss_novel = tf.reduce_mean(tf.square(pi.vpred_novel - ret_novel))

    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]

    total_loss_novel = pol_surr_novel + pol_entpen + vf_loss_novel
    losses_novel = [pol_surr_novel, pol_entpen, vf_loss_novel, meankl, meanent]

    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    policy_var_list = pi.get_trainable_variables(scope='pi/pol')

    policy_var_count = 0
    for vars in policy_var_list:
        count_in_var = 1
        for dim in vars.shape._dims:
            count_in_var *= dim
        policy_var_count += count_in_var

    var_list = pi.get_trainable_variables(scope='pi/pol') + pi.get_trainable_variables(scope='pi/vf/')
    var_list_novel = pi.get_trainable_variables(scope='pi/pol') + pi.get_trainable_variables(scope='pi/vf_novel/')
    var_list_pi = pi.get_trainable_variables(scope='pi/pol') + pi.get_trainable_variables(
        scope='pi/vf/') + pi.get_trainable_variables(scope='pi/vf_novel/')

    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])

    lossandgrad_novel = U.function([ob, ac, atarg_novel, ret_novel, lrmult],
                                   losses_novel + [U.flatgrad(total_loss_novel, var_list_novel)])

    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    adam_novel = MpiAdam(var_list_novel, epsilon=adam_epsilon)
    adam_all = MpiAdam(var_list_pi, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])

    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)
    compute_losses_novel = U.function([ob, ac, atarg_novel, ret_novel, lrmult], losses_novel)

    U.initialize()

    adam.sync()
    adam_novel.sync()
    adam_all.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0

    novelty_update_iter_cycle = 10
    novelty_start_iter = 50
    novelty_update = True

    tstart = time.time()
    lenbuffer = deque(maxlen=100)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100)  # rolling buffer for episode rewards
    rewnovelbuffer = deque(maxlen=100)  # rolling buffer for episode novelty rewards

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0,
                max_seconds > 0]) == 1, "Only one time constraint permitted"

    # This for debug purpose
    # from collections import defaultdict
    # sum_batch = {}
    # sum_batch = defaultdict(lambda: 0, sum_batch)

    while True:
        # if iters_so_far == 5:
        #     print("BREAK PLACEHOLDER")

        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************" % iters_so_far)

        seg = seg_gen.__next__()

        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, atarg_novel, tdlamret, tdlamret_novel = seg["ob"], seg["ac"], seg["adv"], seg["adv_novel"], seg[
            "tdlamret"], seg["tdlamret_novel"]

        vpredbefore = seg["vpred"]  # predicted value function before udpate
        vprednovelbefore = seg['vpred_novel']  # predicted novelty value function before update

        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate
        atarg_novel = (
                              atarg_novel - atarg_novel.mean()) / atarg_novel.std()  # standartized novelty advantage function estimate

        d = Dataset(
            dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, atarg_novel=atarg_novel, vtarg_novel=tdlamret_novel),
            shuffle=not pi.recurrent)

        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

        assign_old_eq_new()  # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        same_update_direction = []  # True
        task_gradient_mag = []
        novel_gradient_mag = []
        # Here we do a bunch of optimization epochs over the data

        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)

                *newlosses_novel, g_novel = lossandgrad_novel(batch["ob"], batch["ac"], batch["atarg_novel"],
                                                              batch["vtarg_novel"],
                                                              cur_lrmult)

                pol_g = g[0:policy_var_count]
                pol_g_novel = g_novel[0:policy_var_count]

                comm = MPI.COMM_WORLD

                pol_g_reduced = np.zeros_like(pol_g)
                pol_g_novel_reduced = np.zeros_like(pol_g_novel)

                comm.Allreduce(pol_g, pol_g_reduced, op=MPI.SUM)
                pol_g_reduced /= comm.Get_size()
                comm.Allreduce(pol_g_novel, pol_g_novel_reduced, op=MPI.SUM)
                pol_g_novel_reduced /= comm.Get_size()

                final_gradient = np.zeros(len(g) + len(g_novel) - policy_var_count)
                final_gradient[policy_var_count::] = np.concatenate(
                    (g[policy_var_count::], g_novel[policy_var_count::]))

                pol_g_normalized = pol_g / np.linalg.norm(pol_g)
                pol_g_novel_normalized = pol_g_novel / np.linalg.norm(pol_g_novel)

                dot = np.dot(pol_g_novel_normalized, pol_g_normalized)

                task_gradient_mag.append(np.linalg.norm(pol_g))
                novel_gradient_mag.append(np.linalg.norm(pol_g_novel))
                same_update_direction.append(dot)

                if (dot > 0):

                    bisector = (pol_g_normalized + pol_g_novel_normalized)
                    bisector_normalized = bisector / np.linalg.norm(bisector)

                    quartersector = (bisector_normalized + pol_g_normalized)
                    quartersector_normalized = quartersector / np.linalg.norm(quartersector)

                    octsector = quartersector+pol_g_normalized
                    octsector_normalized = octsector/np.linalg.norm(octsector)

                    target_dir = octsector_normalized #quartersector_normalized
                    final_gradient[0:policy_var_count] = (np.dot(pol_g_novel, target_dir)+np.dot(pol_g,target_dir))*0.5 * target_dir
                    # final_gradient[0:policy_var_count] = pol_g
                    adam_all.update(final_gradient, optim_stepsize * cur_lrmult)
                    # same_update_direction = True
                else:

                    task_projection = np.dot(pol_g, pol_g_novel_normalized) * pol_g_novel_normalized

                    #novel_projection = np.dot(pol_g_normalized, pol_g_novel) * pol_g_normalized

                    # final_pol_gradient = pol_g_novel - novel_projection
                    final_pol_gradient = pol_g - task_projection

                    final_gradient[0:policy_var_count] = final_pol_gradient

                    # adam_novel.update(final_gradient, optim_stepsize * cur_lrmult)
                    adam_all.update(final_gradient, optim_stepsize * cur_lrmult)

                    # adam.update(final_gradient, optim_stepsize * cur_lrmult)
                    # same_update_direction = False

                # step = optim_stepsize * cur_lrmult

                # adam.update(g, optim_stepsize * cur_lrmult)

                # adam_novel.update(g_novel, optim_stepsize * cur_lrmult)

                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            # newlosses_novel = compute_losses_novel(batch["ob"], batch["ac"], batch["atarg_novel"], batch["vtarg_novel"],
            #                                        cur_lrmult)
            losses.append(newlosses)
        meanlosses, _, _ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg['ep_rets_novel'])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, rews_novel = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        rewnovelbuffer.extend(rews_novel)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpRNoveltyRewMean", np.mean(rewnovelbuffer))

        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        if iters_so_far >= novelty_start_iter and iters_so_far % novelty_update_iter_cycle == 0:
            novelty_update = not novelty_update

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.record_tabular("RelativeDirection", np.array(same_update_direction).mean())
        logger.record_tabular("TaskGradMag", np.array(task_gradient_mag).mean())
        logger.record_tabular("NoveltyGradMag", np.array(novel_gradient_mag).mean())

        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()

    return pi


# def projection_update()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
