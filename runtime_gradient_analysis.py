import matplotlib.pyplot as plot

fig, axs = plot.subplots(3, 2)

num_vars = np.arange(len(pol_g_reduced))
axs[0][0].plot(num_vars, pol_g_reduced, color='b', label='NN Weight of Task gradient')
axs[0][0].set_xlabel('NN Variables')
axs[0][0].set_ylabel('NN Weight Value')
axs[0][0].legend()

axs[0][1].plot(num_vars, pol_g_novel_reduced, color='r', label='NN Weight of Novelty gradient')
axs[0][1].set_xlabel('NN Variables')
axs[0][1].set_ylabel('NN Weight Value')
axs[0][1].legend()

num_vars = np.arange(len(pol_g_novel_reduced_no_noise))
axs[1][0].plot(num_vars, pol_g_reduced_no_noise, color='b', label='NN Weight of Task gradient')
axs[1][0].set_xlabel('NN Variables')
axs[1][0].set_ylabel('NN Weight Value')
axs[1][0].legend()

axs[1][1].plot(num_vars, pol_g_novel_reduced_no_noise, color='r', label='NN Weight of Novelty gradient')
axs[1][1].set_xlabel('NN Variables')
axs[1][1].set_ylabel('NN Weight Value')
axs[1][1].legend()

num_vars = np.arange(len(pol_g_reduced_no_noise[26*64:len(pol_g_reduced_no_noise)-5*64]))
axs[2][0].plot(num_vars, pol_g_reduced_no_noise[26*64:len(pol_g_reduced_no_noise)-5*64], color='b', label='NN Weight of Task gradient')
axs[2][0].set_xlabel('NN Variables')
axs[2][0].set_ylabel('NN Weight Value')
axs[2][0].legend()

axs[2][1].plot(num_vars, pol_g_novel_reduced_no_noise[26*64:len(pol_g_novel_reduced_no_noise)-5*64], color='r', label='NN Weight of Novelty gradient')
axs[2][1].set_xlabel('NN Variables')
axs[2][1].set_ylabel('NN Weight Value')
axs[2][1].legend()

fig.tight_layout()
plot.show()

aa_dot_full = np.dot(pol_g_reduced/np.linalg.norm(pol_g_reduced),pol_g_novel_reduced/np.linalg.norm(pol_g_novel_reduced))
aa_dot_no_noise = np.dot(pol_g_reduced_no_noise/np.linalg.norm(pol_g_reduced_no_noise),pol_g_novel_reduced_no_noise/np.linalg.norm(pol_g_novel_reduced_no_noise))
aa_dot_only_hidden = np.dot(pol_g_novel_reduced_no_noise[26*64:len(pol_g_novel_reduced_no_noise)-5*64]/np.linalg.norm(pol_g_novel_reduced_no_noise[26*64:len(pol_g_novel_reduced_no_noise)-5*64]),pol_g_reduced_no_noise[26*64:len(pol_g_reduced_no_noise)-5*64]/np.linalg.norm(pol_g_reduced_no_noise[26*64:len(pol_g_reduced_no_noise)-5*64]))