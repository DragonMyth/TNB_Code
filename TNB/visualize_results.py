from TNB.Util.post_training_process import *


#
# render_policy('SimplerPathFinding-v0', stoch=False)

# save_dir = collect_rollouts_from_dir('SimplerPathFinding-v2', 1, 'test.pkl', 0,
#                                      policy_gap=5,
#                                      policy_file_basename='policy_params_',
#                                      start_num=100,
#                                      traj_per_policy_per_process=40,
#                                      data_dir='/Users/dragonmyth/Documents/CS/NoveltyEnforcement/data/ppo_SimplerPathFinding-v0_run_2_seed_54/2018-09-27_16:36:55')
# plot_path_data(save_dir)
# plot_path_data()

# render_policy('DartTurtle-v0', stoch=True, record=True)

# autoencoder_dir = "novelty_data/local/autoencoders/"
#
# autoencoder0_name = autoencoder_dir + "DartReacher3d-v1_autoencoder_seed_0_run_0.h5"
# autoencoder1_name = autoencoder_dir + "DartTurtle-v4_autoencoder_seed_0_run_1.h5"
# autoencoder3 = load_model(autoencoder_dir + "SimplerPathFinding-v0_autoencoder_for_run_2_seed_102.h5")
# autoencoder4 = load_model(autoencoder_dir + "new_path_finding_autoencoder_autoencoder_4_seed=54.h5")
# autoencoder_list = []
# render_policy('DartTurtle-v4', stoch=True, record=True, policy_func=mirror_turtle_policy_fn,
#               autoencoder_name_list=autoencoder_list)

# render_policy('MountainCarContinuous-v0', stoch=False,autoencoder_name_list=autoencoder_list)
# # render_policy('DartTurtle-v5', action_skip=5, stoch=True, record=True)


def render_reacher():
    render_policy('DartReacher3d-v1', stoch=False, record=True,
                  autoencoder_name_list=[], num_runs=1)


def render_deceptive_reacher(random=False):
    render_policy('DartReacher3d-v2', stoch=False, record=True,
                  autoencoder_name_list=[], random_policy=random, num_runs=1)


def render_deceptive_maze():
    render_policy('PathFindingDeceptive-v0', stoch=True, record=True,
                  autoencoder_name_list=[], num_runs=4)


def render_four_way_maze():
    render_policy('SimplerPathFinding-v0', stoch=True, record=True,
                  autoencoder_name_list=[], num_runs=1)


def render_hopper():
    render_policy('DartHopper-v2', stoch=False, record=True,
                  autoencoder_name_list=[], random_policy=False)


render_deceptive_reacher()
