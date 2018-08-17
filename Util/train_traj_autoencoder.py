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

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize

environment = 'DartHumanFullSwim-v0'

model_filename = 'new_path_finding_biased_autoencoder_1.h5'

autoencoder = None

openFileOption = {}
# openFileOption['initialdir'] = './data/local/experiment'
filenames = askopenfilenames(**openFileOption)

dataset = []
for file in filenames:
    dataset.extend(joblib.load(file))
dataset = np.array(dataset)
np.random.shuffle(dataset)

model_dim = len(dataset[0][0])
traj_dim = len(dataset[0])

# dqLim = 50 # this works for turtle model
# dqLim = 30

dqLim = 10  # This works for path finding model
qlim = 10


def normalizeTraj(traj, minq, maxq, mindq, maxdq):
    traj[:, :, 0:int(len(traj[0, 0]) / 2)] /= (maxq - minq)
    traj[:, :, int(len(traj[0, 0]) / 2)::] /= (maxdq - mindq)
    return traj


def invNormalizeTraj(traj, minq, maxq, mindq, maxdq):
    traj[:, 0:int(len(traj[0]) / 2)] *= (maxq - minq)
    traj[:, int(len(traj[0]) / 2)::] *= (maxdq - mindq)
    return traj


dataset = normalizeTraj(dataset, -qlim, qlim, -dqLim, dqLim)

dataset = dataset.reshape((len(dataset), np.prod(dataset.shape[1:])))

# norm_dataset = sklearn.preprocessing.normalize(dataset, norm="l2", return_norm=True)



# data_norms = norm_dataset[1]

x_train = None
x_test = None
# sample = dataset[18000::]

if not os.path.isfile('./AutoEncoder/local/' + model_filename):
    x_train = dataset[:int(len(dataset) / 5.0 * 4)]
    x_test = dataset[int(len(dataset) / 5.0 * 4)::]
    input_dim = len(x_train[0])

    # Build FCN

    encoding_dim = 64

    input_traj = Input(shape=(input_dim,))
    encoded = Dense(512, activation='relu')(input_traj)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    # encoded = Dense(32,activation='relu')(encoded)
    # #
    # decoded = Dense(64,activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='tanh')(decoded)

    autoencoder = Model(input_traj, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    history = autoencoder.fit(x_train, x_train,
                              validation_data=(x_test, x_test),
                              epochs=300,
                              batch_size=1024,
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

    encoded_input = Input(shape=(encoding_dim,))
    decoded_layer = autoencoder.layers[-1]

    autoencoder.name = model_filename
    autoencoder.save('../AutoEncoder/local/' + model_filename)
else:

    x_test = dataset[int(len(dataset) / 5.0 * 4)::]
    autoencoder = load_model("../AutoEncoder/local/" + model_filename)


def vis_in_graph():
    decoded_traj = autoencoder.predict(x_test)
    n = 13  # how many digits we will display
    gap = 13
    plt.figure(figsize=(40, 6))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        img = (x_test[i * gap] ** 2).reshape(traj_dim, model_dim)
        plt.imshow(img)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow((decoded_traj[i * gap] ** 2).reshape(traj_dim, model_dim))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display Reconstruction Difference
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(((decoded_traj[i * gap] - x_test[i * gap]) ** 2).reshape(traj_dim, model_dim))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def vis_in_motion():
    print("Motion Visualization")
    env = normalize(GymEnv(environment, record_video=True, log_dir='./AutoEncoder/AutoencoderVis'))

    # This following traj is the original one

    # This following traj is the reconstructed one
    # traj = (autoencoder.predict(x_test)[0]*data_norms[0]).reshape(15,44)
    action_dim = env.action_dim
    horizon = env.horizon
    continue_length = int(horizon / traj_dim)
    trials_nums = 1
    for _ in range(trials_nums):
        env.reset()
        random_trial = random.randint(0, len(x_test) - 1)

        # traj = (x_test[random_trial] * data_norms[random_trial]).reshape(traj_dim, model_dim)
        traj_1 = np.copy(x_test)[random_trial].reshape(traj_dim, model_dim)[:]
        traj_1 = invNormalizeTraj(traj_1, -np.pi, np.pi, -dqLim, dqLim)

        for step in range(horizon):

            if step % continue_length == 0:
                if int(step / continue_length) < traj_dim:
                    state = traj_1[int(step / continue_length)]

                    # These are for humanoid swimmer
                    # q = np.concatenate([[0] * 6, state[0:2], state[4:24]])
                    # dq = np.concatenate([[0] * 6, state[2:4], state[24::]])

                    q = np.concatenate([[0] * 6, state[0:int(model_dim / 2)]])
                    dq = np.concatenate([[0] * 6, state[int(model_dim / 2)::]])

                    skeleton = env.wrapped_env.env.env.env.robot_skeleton

                    skeleton.set_positions(q)
                    skeleton.set_velocities(dq)

            action = np.array([0] * action_dim)
            env.step(action)
            env.render()

        env.reset()
        # traj = (autoencoder.predict(x_test)[random_trial] * data_norms[random_trial]).reshape(traj_dim, model_dim)
        traj_2 = autoencoder.predict(np.array([np.copy(x_test)[random_trial]])).reshape(traj_dim, model_dim)[:]
        traj_2 = invNormalizeTraj(traj_2, -np.pi, np.pi, -dqLim, dqLim)

        for step in range(horizon):

            if step % continue_length == 0:
                if int(step / continue_length) < traj_dim:
                    state = traj_2[int(step / continue_length)]
                    # q = np.concatenate([[0] * 6, state[0:2], state[4:24]])
                    # dq = np.concatenate([[0] * 6, state[2:4], state[24::]])

                    q = np.concatenate([[0] * 6, state[0:int(model_dim / 2)]])
                    dq = np.concatenate([[0] * 6, state[int(model_dim / 2)::]])

                    skeleton = env.wrapped_env.env.env.env.robot_skeleton

                    skeleton.set_positions(q)
                    skeleton.set_velocities(dq)

            action = np.array([0] * action_dim)
            env.step(action)
            env.render()


def plot_reconst_err():
    decoded_traj = autoencoder.predict(x_test)

    items = np.arange(len(x_test))

    diff = decoded_traj[:] - x_test[:]

    l2_norm = np.linalg.norm(diff[:], axis=1)

    plt.plot(items, l2_norm)
    plt.show()
    print(l2_norm.shape)
    print(np.mean(l2_norm))


vis_in_graph()
# vis_in_motion()
plot_reconst_err()
print("Done")
