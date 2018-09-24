import random

import joblib
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
    def __init__(self, qnorm, dqnorm, model_name, input_dim, n_rows, n_cols):
        self.qnorm = qnorm
        self.dqnorm = dqnorm
        self.model_name = model_name
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.input_dim = input_dim
        self.autoencoder = self.build_autoencoder()

    def normalizeTraj(self, traj):
        traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (self.qnorm)
        traj[:, :, int(len(traj[0, 0]) / 2)::] /= (self.dqnorm)
        return traj

    def invNormalizeTraj(self, traj):
        traj[:, 0:int(len(traj[0]) / 2)] *= (self.qnorm)
        traj[:, int(len(traj[0]) / 2)::] *= (self.dqnorm)
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

        decoded = Dense(self.input_dim, activation='tanh')(decoded)

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
        plt.show()

        # encoded_input = Input(shape=(encoding_dim,))
        # decoded_layer = self.autoencoder.layers[-1]

        self.autoencoder.name = model_filename
        self.autoencoder.save('../novelty_data/local/autoencoders/' + model_filename)

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
            plt.imshow(img)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow((decoded_traj[i * gap] ** 2).reshape(self.n_rows, self.n_cols))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display Reconstruction Difference
            ax = plt.subplot(3, n, i + 1 + 2 * n)
            plt.imshow(((decoded_traj[i * gap] - x_test[i * gap]) ** 2).reshape(self.n_rows, self.n_cols))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def plot_reconst_err(self, x_test):
        decoded_traj = self.autoencoder.predict(x_test)

        items = np.arange(len(x_test))

        diff = decoded_traj[:] - x_test[:]

        l2_norm = np.linalg.norm(diff[:], axis=1)

        plt.plot(items, l2_norm)
        plt.show()
        print(l2_norm.shape)
        print(np.mean(l2_norm))


if __name__ == '__main__':

    model_filename = 'new_path_finding_autoencoder_autoencoder_1.h5'

    openFileOption = {}
    # openFileOption['initialdir'] = './data/local/experiment'
    filenames = askopenfilenames(**openFileOption)

    dataset = []
    for file in filenames:
        dataset.extend(joblib.load(file))
    dataset = np.array(dataset)

    model_dim = len(dataset[0][0])
    traj_dim = len(dataset[0])

    autoencoder = Autoencoder(10, 10, model_filename, model_dim * traj_dim, model_dim, traj_dim)

    dataset = autoencoder.normalizeTraj(dataset)
    dataset = dataset.reshape((len(dataset), np.prod(dataset.shape[1:])))

    x_train = dataset[:int(len(dataset) / 5.0 * 4)]
    x_test = dataset[int(len(dataset) / 5.0 * 4)::]

    autoencoder.train(x_train, x_test, epoches=3, batchsize=1024)
    autoencoder.vis_in_graph(x_test)
    autoencoder.plot_reconst_err(x_test)
    print("Done")
