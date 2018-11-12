import numpy as np
import matplotlib.pyplot as plt

dim = 2000
mean = 0
std = 0.000001
num_samples = 1000
dots = []
all_a = []
all_b = []
for i in range(num_samples):
    a = np.random.normal(mean, std, dim)
    b = np.random.normal(mean, std, dim)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    all_a.append(a)
    all_b.append(b)

    dot = np.dot(a, b)
    dots.append(dot)

plt.figure()
plt.hist(dots, stacked=True, bins=30)
plt.xlabel('Relative Directions')
plt.ylabel('Probability')
plt.legend()
plt.yscale('linear')
plt.xscale('linear')
plt.show()

fig, axs = plt.subplots(2, 1)

num_vars = np.arange(dim)
axs[0].plot(num_vars, all_a[0], color='b')

axs[1].plot(num_vars, all_b[0], color='r')
# plot.yscale('linear')
# plot.xscale('linear')
fig.tight_layout()
plt.show()
