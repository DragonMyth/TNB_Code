from matplotlib import pyplot as plot

import csv
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import joblib

openFileOption = {}
openFileOption['initialdir'] = './data/local/experiment'

# filename = askopenfilename(**openFileOption)
directory = askdirectory(**openFileOption)

snapshot_dir = directory  # filename[0:filename.rfind('/') + 1]
category_idx_lookup = dict()
data_lookup = dict()
progress_filename = directory + '/progress.csv'
with open(progress_filename, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0
    for row in reader:
        if i == 0:
            i += 1
            for j in range(len(row)):
                category_idx_lookup[row[j]] = j
                data_lookup[j] = []
            # category_idx_lookup['Iteration'] = len(row)
            # data_lookup[len(row)] = []
            continue
        if row:
            for j in range(len(row)):
                # if(row[j])
                if row[j] != 'True' and row[j] != 'False':
                    data_lookup[j].append(float(row[j]))
            # data_lookup[len(row)].append(i - 1)
gradient_filename = directory + '/gradientinfo.pkl'
try:
    gradient_info = joblib.load(gradient_filename)
except(FileNotFoundError):
    gradient_info = None
print("File Loaded!")


def plot_progress():
    EpisodesSofar = np.array(data_lookup[category_idx_lookup['EpisodesSoFar']])
    ItersSofar = np.arange(len(EpisodesSofar))
    indices = [i for i, x in enumerate(EpisodesSofar) if x == 0]

    AverageReturns = np.array(data_lookup[category_idx_lookup['EpRewMean']])
    EpRNoveltyRewMean = np.array(data_lookup[category_idx_lookup['EpRNoveltyRewMean']])
    RelativeDirection = np.array(data_lookup[category_idx_lookup['RelativeDirection']])
    # TaskGradientMag = np.array(data_lookup[category_idx_lookup['TaskGradMag']])
    # NoveltyGradientMag = np.array(data_lookup[category_idx_lookup['NoveltyGradMag']])
    # MaxReturns = data_lookup[category_idx_lookup['MaxReturn']]
    # MinReturns = data_lookup[category_idx_lookup['MinReturn']]
    fig, axs = plot.subplots(2, 1)
    # plot.figure()
    # print(AverageReturns)
    axs[0].plot(ItersSofar, AverageReturns, 'r', label='Average Return')

    # plot.plot(Iterations[begin:end], MinReturns[begin:end], 'g', label='Minimum Return')
    # plot.plot(Iterations[begin:end], MaxReturns[begin:end], 'b', label='Maximum Return')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Expected Return')
    # plot.title('Mirror Enforcement Larger Loss')
    # plot.yscale('symlog')
    axs[0].legend()
    axs[0].set_yscale('linear')
    axs[0].set_xscale('linear')

    # plot.savefig(snapshot_dir + '/TaskReturnTrainingCurve' + '.jpg')
    # plot.show()
    #
    # plot.figure()

    axs[1].plot(ItersSofar, EpRNoveltyRewMean, 'b', label='Average Novelty Return')

    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Expected Return')

    axs[1].legend()
    # axs[1].set_yscale('symlog', linthreshy=1e-5)
    axs[1].set_yscale('linear')
    axs[1].set_xscale('linear')

    # plot.legend()
    fig.tight_layout()

    plot.savefig(snapshot_dir + '/ReturnTrainingCurve' + '.jpg')

    plot.show()

    plot.figure()

    plot.hist(RelativeDirection, stacked=True, bins=30)
    plot.xlabel('Relative Directions')
    plot.ylabel('Probability')

    plot.legend()
    plot.yscale('linear')
    plot.xscale('linear')
    # plot.legend()
    plot.savefig(snapshot_dir + '/relative_direction_prob' + '.jpg')

    plot.show()
    #
    # plot.figure()
    #
    # plot.plot(ItersSofar, TaskGradientMag, 'r', label='Task Gradient Magnitudes')
    # plot.plot(ItersSofar, NoveltyGradientMag, 'b', label='Novelty Gradient Magnitudes')
    #
    # plot.xlabel('Iterations')
    # plot.ylabel('Gradient Magnitudes')
    #
    # plot.legend()
    # plot.yscale('linear')
    # plot.xscale('linear')
    # plot.legend()
    # plot.savefig(snapshot_dir + '/gradient_magnitudes' + '.jpg')
    #
    # plot.show()


def plot_gradient():
    task_gradients = gradient_info['task_gradients']
    novelty_gradients = gradient_info['novelty_gradients']
    RelativeDirection = np.array(data_lookup[category_idx_lookup['RelativeDirection']])

    task_gradients_normalized = task_gradients / np.linalg.norm(task_gradients, axis=1, keepdims=True)
    novelty_gradients_normalized = novelty_gradients / np.linalg.norm(novelty_gradients, axis=1, keepdims=True)

    #
    dot_products = np.zeros(len(task_gradients))
    for i in range(len(task_gradients)):
        task_norm = task_gradients[i] / np.linalg.norm(task_gradients[i])
        novel_norm = novelty_gradients[i] / np.linalg.norm(novelty_gradients[i])
        dot_products[i] = np.dot(task_norm, novel_norm)

    iters = np.arange(len(task_gradients))
    plot.figure()
    plot.plot(iters, dot_products)
    plot.show()

    max_relative_direction_timestep = np.argmax(RelativeDirection)

    task_gradient_max = task_gradients[max_relative_direction_timestep] / np.linalg.norm(
        task_gradients[max_relative_direction_timestep])
    novelty_gradient_max = novelty_gradients[max_relative_direction_timestep] / np.linalg.norm(
        novelty_gradients[max_relative_direction_timestep])
    # task_gradient_max = task_gradient_max / np.linalg.norm(task_gradient_max)
    # novelty_gradient_max = novelty_gradient_max / np.linalg.norm(novelty_gradient_max)

    fig, axs = plot.subplots(2, 1)

    num_vars = np.arange(len(task_gradient_max[:]))
    axs[0].plot(num_vars, task_gradient_max[:], color='b', label='NN Weight of Task gradient')
    axs[0].set_xlabel('NN Variables')
    axs[0].set_ylabel('NN Weight Value')
    axs[0].legend()

    axs[1].plot(num_vars, novelty_gradient_max[:], color='r', label='NN Weight of Novelty gradient')
    axs[1].set_xlabel('NN Variables')
    axs[1].set_ylabel('NN Weight Value')
    axs[1].legend()

    print(np.dot(novelty_gradient_max, task_gradient_max))
    print(RelativeDirection[max_relative_direction_timestep])

    # plot.yscale('linear')
    # plot.xscale('linear')
    fig.tight_layout()
    plot.show()

    print("Max Relative Direction at step: ", max_relative_direction_timestep)
    print("Relative Direction is: ", RelativeDirection[max_relative_direction_timestep])
    print("Gradient magnitude verify: ", np.linalg.norm(task_gradient_max), np.linalg.norm(novelty_gradient_max))
    print("Relative Direction verify: ", np.dot(task_gradient_max, novelty_gradient_max))


plot_progress()
# plot_gradient()
