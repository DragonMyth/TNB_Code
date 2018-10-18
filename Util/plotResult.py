from matplotlib import pyplot as plot

import csv
import numpy as np
from tkinter.filedialog import askopenfilename

openFileOption = {}
openFileOption['initialdir'] = './data/local/experiment'
filename = askopenfilename(**openFileOption)
snapshot_dir = filename[0:filename.rfind('/') + 1]
category_idx_lookup = dict()
data_lookup = dict()
with open(filename, newline='') as csvfile:
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

EpisodesSofar = np.array(data_lookup[category_idx_lookup['EpisodesSoFar']])
ItersSofar = np.arange(len(EpisodesSofar))
indices = [i for i, x in enumerate(EpisodesSofar) if x == 0]

AverageReturns = np.array(data_lookup[category_idx_lookup['EpRewMean']])
EpRNoveltyRewMean = np.array(data_lookup[category_idx_lookup['EpRNoveltyRewMean']])
RelativeDirection = np.array(data_lookup[category_idx_lookup['RelativeDirection']])
TaskGradientMag = np.array(data_lookup[category_idx_lookup['TaskGradMag']])
NoveltyGradientMag = np.array(data_lookup[category_idx_lookup['NoveltyGradMag']])
# MaxReturns = data_lookup[category_idx_lookup['MaxReturn']]
# MinReturns = data_lookup[category_idx_lookup['MinReturn']]

plot.figure()
# print(AverageReturns)
plot.plot(ItersSofar, AverageReturns, 'r', label='Average Return')
plot.plot(ItersSofar, EpRNoveltyRewMean, 'b', label='Average Novelty Return')

# plot.plot(Iterations[begin:end], MinReturns[begin:end], 'g', label='Minimum Return')
# plot.plot(Iterations[begin:end], MaxReturns[begin:end], 'b', label='Maximum Return')
plot.xlabel('Iterations')
plot.ylabel('Expected Return')
# plot.title('Mirror Enforcement Larger Loss')
# plot.yscale('symlog')
plot.legend()
plot.yscale('symlog')
plot.xscale('linear')

plot.savefig(snapshot_dir + '/progress' + '.jpg')
plot.show()

plot.figure()

plot.plot(ItersSofar, RelativeDirection, 'r', label='Relative Direction')
plot.xlabel('Iterations')

plot.legend()
plot.yscale('linear')
plot.xscale('linear')

# plot.legend()
plot.savefig(snapshot_dir + '/progress_2' + '.jpg')

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

plot.figure()

plot.plot(ItersSofar, TaskGradientMag, 'r', label='Task Gradient Magnitudes')
plot.plot(ItersSofar, NoveltyGradientMag, 'b', label='Novelty Gradient Magnitudes')

plot.xlabel('Iterations')
plot.ylabel('Gradient Magnitudes')

plot.legend()
plot.yscale('linear')
plot.xscale('linear')
# plot.legend()
plot.savefig(snapshot_dir + '/gradient_magnitudes' + '.jpg')

plot.show()
