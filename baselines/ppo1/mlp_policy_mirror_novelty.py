from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
from scipy import sparse

import numpy as np


class MlpPolicyMirrorNovelty(object):
    recurrent = False

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, mirror_loss=False,
              observation_permutation=[], action_permutation=[]):

        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        if mirror_loss:
            # construct permutation matrices
            obs_perm_mat = np.zeros((len(observation_permutation), len(observation_permutation)), dtype=np.float32)
            act_perm_mat = np.zeros((len(action_permutation), len(action_permutation)), dtype=np.float32)

            for i, perm in enumerate(observation_permutation):
                obs_perm_mat[i][perm] = 1
            for i, perm in enumerate(action_permutation):
                act_perm_mat[i][perm] = 1
            # obs_perm_mat = sparse.csc_matrix(obs_perm_mat)
            # act_perm_mat = sparse.csc_matrix(act_perm_mat)
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i" % (i + 1),
                                                      kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:, 0]

        with tf.variable_scope('vf_novel'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i" % (i + 1),
                                                      kernel_initializer=U.normc_initializer(1.0)))
            self.vpred_novel = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[
                               :, 0]

        params = []

        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                # last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i' % (i + 1),
                #                                       kernel_initializer=U.normc_initializer(1.0)))

                rt, pw, pb = U.dense_wparams(last_out, hid_size, name='fc%i' % (i + 1))
                last_out = tf.nn.tanh(rt)

                params.append([pw, pb])

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):

                # mean_orig = tf.layers.dense(last_out, pdtype.param_shape()[0] // 2, name='final',
                #                             kernel_initializer=U.normc_initializer(1.0))

                mean_orig, pw, pb = U.dense_wparams(last_out, pdtype.param_shape()[0] // 2, name="final",
                                                    weight_init=U.normc_initializer(1))
                params.append([pw, pb])

                # self.mean = mean
                mean = mean_orig
                if mirror_loss:

                    mirrored_obz = tf.matmul(obz, obs_perm_mat)
                    last_mirrored_out = mirrored_obz
                    for i in range(len(params) - 1):
                        last_mirrored_out = tf.nn.tanh(tf.matmul(last_mirrored_out, params[i][0]) + params[i][1])

                    last_mirrored_out = tf.matmul(last_mirrored_out, params[-1][0]) + params[-1][1]

                    mirr_mean = tf.matmul(last_mirrored_out, act_perm_mat)
                    # self.mirr_mean = mirr_mean
                    mean = tf.divide(tf.add(mean_orig, mirr_mean), 2)

                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                         initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final',
                                          kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.get_pd_param = U.function([], [logstd])

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function([stochastic, ob], [ac, self.vpred, self.vpred_novel])

        self._orig_mean = U.function([ob], [mean_orig])
        self._mirr_mean = U.function([ob], [mirr_mean])
        self._mirr_obs_mean = U.function([ob], [last_mirrored_out])

    def act(self, stochastic, ob):
        ac1, vpred1, vpred_novel = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0], vpred_novel[0]

    def log_std(self, ob):
        log_std = self.get_pd_param()
        return log_std

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self, scope=None):
        if scope is None:
            scope = self.scope
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

    def get_initial_state(self):
        return []
