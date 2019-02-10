from matplotlib import pyplot as plot

import csv
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import joblib
import os

openFileOption = {}
openFileOption['initialdir'] = '.'

# filename = askopenfilename(**openFileOption)
directory = askdirectory(**openFileOption)

env_directory = directory  # filename[0:filename.rfind('/') + 1]

seed_dir_list = os.listdir(env_directory)
# for dir in dir_list:
#
print(seed_dir_list)

#

learning_data_all_seeds = []

for seed_dir in seed_dir_list:
    if (seed_dir[0] == '.'):
        continue
    learning_data_all_runs = []
    run_dir_list = os.listdir(env_directory + '/' + seed_dir)
    for run_dir in run_dir_list:
        if (run_dir[0] == '.'):
            continue
        category_idx_lookup = dict()
        data_lookup = dict()
        progress_filename = directory + '/' + seed_dir + '/' + run_dir + '/policy/progress.csv'
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
        learning_data_all_runs.append((category_idx_lookup, data_lookup))
    learning_data_all_seeds.append(learning_data_all_runs)
print("Files Loaded!")

# def sanitizd_return_data():
task_rts_for_all_seeds = []
novelty_rts_for_all_seeds = []
returns_for_all_seeds = []
for learning_for_all_runs in learning_data_all_seeds:
    # num_runs = len(learning_data_all_runs)
    task_rwds = []
    novelty_rwds = []

    for learning_data in learning_for_all_runs:
        category_idx_lookup = learning_data[0]
        data_lookup = learning_data[1]

        task_rwd = np.array(data_lookup[category_idx_lookup['EpRewMean']])
        novelty_rwd = np.array(data_lookup[category_idx_lookup['EpRNoveltyRewMean']])
        task_rwds.append(task_rwd)
        novelty_rwds.append(novelty_rwd)

    min_len = len(task_rwds[0])
    for rwd in task_rwds:
        if (len(rwd) < min_len):
            min_len = len(rwd)
    for i in range(len(task_rwds)):
        task_rwds[i] = task_rwds[i][:min_len]
        novelty_rwds[i] = novelty_rwds[i][:min_len]
    task_rwds = np.array(task_rwds)
    novelty_rwds = np.array(novelty_rwds)

    mean_task_rwd = task_rwds.mean(axis=0)
    mean_novelty_rwd = novelty_rwds.mean(axis=0)
    # task_rts_for_all_seeds.append(mean_task_rwd)
    # novelty_rts_for_all_seeds.append(mean_novelty_rwd)
    returns_for_all_seeds.append((mean_task_rwd, mean_novelty_rwd))

fig, axs = plot.subplots(2, 1)
# plot.figure()
# print(AverageReturns)

for seed in returns_for_all_seeds:
    ItersSofar = np.arange(len(seed[0]))

    axs[0].plot(ItersSofar, seed[0])
    axs[1].plot(ItersSofar, seed[1])

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

# axs[1].plot(ItersSofar, EpRNoveltyRewMean, 'b', label='Average Novelty Return')

axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Expected Return')

axs[1].legend()
axs[1].set_yscale('symlog', linthreshy=1e-5)
# axs[1].set_yscale('linear')
axs[1].set_xscale('linear')

# plot.legend()
fig.tight_layout()

# plot.savefig(snapshot_dir + '/ReturnTrainingCurve' + '.jpg')

plot.show()
