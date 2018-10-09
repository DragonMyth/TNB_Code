from Util.post_training_process import *

#
# render_policy('SimplerPathFinding-v0',stoch=False)

# save_dir = collect_rollouts_from_dir('SimplerPathFinding-v2', 1, 'test.pkl', 0,
#                                      policy_gap=5,
#                                      policy_file_basename='policy_params_',
#                                      start_num=100,
#                                      traj_per_policy_per_process=40,
#                                      data_dir='/Users/dragonmyth/Documents/CS/NoveltyEnforcement/data/ppo_SimplerPathFinding-v0_run_2_seed_54/2018-09-27_16:36:55')
# plot_path_data(save_dir)
# plot_path_data()

# render_policy('DartTurtle-v0', stoch=True, record=True)
render_policy('DartTurtle-v4', stoch=True, record=False)
