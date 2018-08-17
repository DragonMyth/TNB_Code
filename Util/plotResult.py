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
            continue
        if row:
            for j in range(len(row)):
                data_lookup[j].append(round(float(row[j]), 5))

Iterations = data_lookup[category_idx_lookup['Iteration']]
indices = [i for i, x in enumerate(Iterations) if x == 0]

AverageReturns = data_lookup[category_idx_lookup['AverageReturn']]
MaxReturns = data_lookup[category_idx_lookup['MaxReturn']]
MinReturns = data_lookup[category_idx_lookup['MinReturn']]

AverageReturns = np.clip(AverageReturns, -35000, 35000)
MaxReturns = np.clip(MaxReturns, -35000, 35000)
MinReturns = np.clip(MinReturns, -35000, 35000)

img_id = 0
for i in range(len(indices)):
    begin = indices[i]

    if (i < len(indices) - 1):
        end = indices[i + 1]
    else:
        end = -1
    plot.figure()

    plot.plot(Iterations[begin:end], AverageReturns[begin:end], 'r', label='Average Return')
    plot.plot(Iterations[begin:end], MinReturns[begin:end], 'g', label='Minimum Return')
    plot.plot(Iterations[begin:end], MaxReturns[begin:end], 'b', label='Maximum Return')
    plot.xlabel('Iterations')
    plot.ylabel('Expected Return')
    # plot.title('Mirror Enforcement Larger Loss')
    # plot.yscale('symlog')
    plot.legend()

    plot.savefig(snapshot_dir + '/progress_' + str(img_id) + '.jpg')
    plot.figure()

    # MirroredLoss = data_lookup[category_idx_lookup['MirroredLoss']]
    # plot.plot(Iterations[begin:end],MirroredLoss[begin:end],label='Mirrored Loss')
    # plot.title('Mirror Enforcement Larger Loss')
    #
    # plot.legend()

    plot.savefig(snapshot_dir + '/mirror_loss_' + str(img_id) + '.jpg')
    img_id += 1

plot.show()
