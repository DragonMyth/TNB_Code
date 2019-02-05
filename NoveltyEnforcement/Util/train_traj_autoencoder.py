import random

import joblib
import matplotlib.colors
import numpy as np
import scipy.misc
import sklearn.preprocessing
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import h5py
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
import os.path
import matplotlib.pyplot as plt


class Autoencoder():
    def __init__(self, qnorm, dqnorm, norm_scales, model_name, input_dim, n_rows, n_cols, save_dir):
        self.qnorm = qnorm
        self.dqnorm = dqnorm
        self.model_name = model_name
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.input_dim = input_dim
        self.save_dir = save_dir
        self.norm_scales = self.generateNormScaleArr(norm_scales)
        # print(self.norm_scales)
        self.autoencoder = self.build_autoencoder()

    def generateNormScaleArr(self, norm_scales):
        norms = np.ones(self.n_cols)
        cur_idx = 0
        for i in range(0, len(norm_scales), 2):
            num_repeat = int(norm_scales[i])
            norms[cur_idx:cur_idx + num_repeat] = norm_scales[i + 1]
            cur_idx += num_repeat
        return norms

    def normalizeTraj(self, traj):

        # traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (self.qnorm)
        # traj[:, :, int(len(traj[0, 0]) / 2)::] /= (self.dqnorm)

        traj[:, :, :] /= self.norm_scales

        return traj

    def invNormalizeTraj(self, traj):
        # traj[:, 0:int(len(traj[0]) / 2)] *= (self.qnorm)
        # traj[:, int(len(traj[0]) / 2)::] *= (self.dqnorm)
        traj[:, :, :] *= self.norm_scales
        return traj

    def build_autoencoder(self):
        encoding_dim = 64

        input_traj = Input(shape=(self.input_dim,))

        encoded = Dense(1024, activation='relu')(input_traj)
        encoded = Dense(512, activation='relu')(encoded)
        encoded = Dense(256, activation='relu')(encoded)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(512, activation='relu')(decoded)
        decoded = Dense(1024, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)

        autoencoder = Model(input_traj, decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        return autoencoder

    def train(self, x_train, x_test, epoches, batchsize):
        history = self.autoencoder.fit(x_train, x_train,
                                       validation_data=(x_test, x_test),
                                       epochs=epoches,
                                       batch_size=batchsize,
                                       shuffle=True,
                                       )
        self.autoencoder.name = model_filename
        # self.autoencoder.save('./novelty_data/local/autoencoders/' + model_filename)
        self.autoencoder.save(self.save_dir + '/' + model_filename)
        # validation_data=(x_test, x_test))
        # list all data in history
        print(history.history.keys())
        # # summarize history for accuracy
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        # # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig('./novelty_data/local/autoencoders/plots/' + model_filename + '_training_curve.png')
        plt.savefig(self.save_dir + '/plots/' + model_filename + '_training_curve.png')

        plt.close()
        # encoded_input = Input(shape=(encoding_dim,))
        # decoded_layer = self.autoencoder.layers[-1]

        # self.autoencoder.name = model_filename
        # self.autoencoder.save('./novelty_data/local/autoencoders/' + model_filename)
        # self.autoencoder.save(self.save_dir + '/' + model_filename)

    # def vis_data_in_motion():

    def vis_in_graph(self, x_test):
        decoded_traj = self.autoencoder.predict(x_test)
        n = 13  # how many digits we will display
        gap = 13
        plt.figure(figsize=(40, 20))
        for i in range(n):
            # display original

            ax = plt.subplot(3, n, i + 1)
            img = (x_test[i * gap] ** 2).reshape(self.n_rows, self.n_cols)
            # norm = matplotlib.colors.SymLogNorm(np.min(abs(img)), linscale=np.min(abs(img)))
            norm = matplotlib.colors.Normalize(vmin=np.min(img), vmax=np.max(img))
            plt.imshow(img, norm=norm, aspect='auto')
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + n)
            img_recon = (decoded_traj[i * gap] ** 2).reshape(self.n_rows, self.n_cols)
            # norm = matplotlib.colors.SymLogNorm(np.min(abs(img_recon)), linscale=np.min(abs(img)))
            plt.imshow(img_recon, norm=norm, aspect='auto')
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            # display Reconstruction Difference
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, i + 1 + 2 * n)

            img_diff = ((decoded_traj[i * gap] - x_test[i * gap]) ** 2).reshape(self.n_rows, self.n_cols)
            # norm = matplotlib.colors.SymLogNorm(np.min(abs(img_diff)), linscale=np.min(abs(img)))

            plt.imshow(img_diff, norm=norm, aspect='auto')
            # plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # plt.savefig('./novelty_data/local/autoencoders/plots/' + model_filename + '_reconstruction_vis.png')
        plt.savefig(self.save_dir + '/plots/' + model_filename + '_reconstruction_vis.png')

        plt.close()
        # plt.show()

    def plot_reconst_err(self, x_test):
        decoded_traj = self.autoencoder.predict(x_test)

        items = np.arange(len(x_test))

        diff = decoded_traj[:] - x_test[:]

        l2_norm = np.linalg.norm(diff[:], axis=1)

        plt.plot(items, l2_norm)
        # plt.savefig('./novelty_data/local/autoencoders/plots/' + model_filename + '_reconstruction_err.png')
        plt.savefig(self.save_dir + '/plots/' + model_filename + '_reconstruction_err.png')

        plt.close()
        # plt.show()
        print(l2_norm.shape)
        print(np.mean(l2_norm))


if __name__ == '__main__':
    # openFileOption = {}
    # openFileOption['initialdir'] = './data/local/experiment'
    # filenames = askopenfilenames(**openFileOption)
    #
    # dataset = []
    # for file in filenames:
    #     dataset.extend(joblib.load(file))
    # dataset = np.array(dataset)
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env', help='environment ID', default='DartReacher3d-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--data_collect_env', help='Environment used to collect data', default='DartReacher3d-v1')
    parser.add_argument('--curr_run', help='Current number of runs in the sequence', default=0)
    parser.add_argument('--qnorm', help='Scale of the q in the skel', default=np.pi)
    parser.add_argument('--dqnorm', help='Scale of the dq in the skel', default=50)
    parser.add_argument('--num_epoch', help='Scale of the dq in the skel', default=300)
    parser.add_argument('--batch_size', help='Scale of the dq in the skel', default=1024)
    parser.add_argument('--norm_scales', help='List of scales used to normalize autoencoder input', nargs='*',
                        default=[])
    parser.add_argument('--run_dir', help='Directory for saving the log files for the large run')

    args = parser.parse_args()

    if not os.path.isdir(args.run_dir + '/autoencoders'):
        os.makedirs(args.run_dir + '/autoencoders')
        os.makedirs(args.run_dir + '/autoencoders/plots')

    norm_scales = np.array(args.norm_scales, dtype=np.float32)

    model_filename = args.env + '_autoencoder_seed_' + str(args.seed) + '_run_' + str(args.curr_run) + '.h5'

    # collected_data_filename = 'novelty_data/local/sampled_paths/' + args.data_collect_env + '_seed_' + str(
    #     args.seed) + '_run_' + str(
    #     args.curr_run) + '.pkl'

    collected_data_filename = args.run_dir + '/sampled_paths/' + args.data_collect_env + '_seed_' + str(
        args.seed) + '_run_' + str(
        args.curr_run) + '.pkl'

    dataset = joblib.load(collected_data_filename)
    dataset = np.array(dataset)

    model_dim = len(dataset[0][0])
    traj_dim = len(dataset[0])

    autoencoder = Autoencoder(float(args.qnorm), float(args.dqnorm), norm_scales, model_filename,
                              model_dim * traj_dim, traj_dim,
                              model_dim, args.run_dir + '/autoencoders')

    dataset = autoencoder.normalizeTraj(dataset)
    dataset = dataset.reshape((len(dataset), np.prod(dataset.shape[1:])))

    x_train = dataset[:int(len(dataset) / 5.0 * 4)]
    x_test = dataset[int(len(dataset) / 5.0 * 4)::]

    autoencoder.train(x_train, x_test, epoches=int(args.num_epoch), batchsize=int(args.batch_size))
    autoencoder.vis_in_graph(x_test)
    autoencoder.plot_reconst_err(x_test)
    print("Done")
