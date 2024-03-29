from utilities.Config import Config
from utilities.rl_utils import get_state_shape
from settings import WORLD_STATE

x, y = get_state_shape(WORLD_STATE)
###################
# Training Params #
###################

config = Config()
config.seed = 4
config.num_episodes_to_run = 100
config.results_filepath = "/polyaxon-exp-output/sidney_tio/"
#config.file_to_save_results_graph = "results/data_and_graphs/Cart_Pole_Results_Graph.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = True
config.model_path = "/polyaxon-exp-output/sidney_tio/policy_2nd.pth"
config.train = False
config.reward_horizon = 1e6
config.old_policy_path = None

##############
# PPO Params #
##############
config.hyperparameters = {
    "learning_rate": 1e-3,
    "linear_hidden_units": [20, 20],
    "final_layer_activation": "SOFTMAX",
    "learning_iterations_per_round": 8,
    "discount_rate": 0.99,
    "batch_norm": False,
    "clip_epsilon": 0.1,
    "episodes_per_learning_round": 12,
    "normalise_rewards": True,
    "gradient_clipping_norm": 0.1,
    "epsilon_decay_rate_denominator": 1000,
    "clip_rewards": False,
    "action_space": 15,
    "obs_space": None,
    "random_policy": True,
    "nn_params": {'NUM_HIDDEN_LAYERS': 3,
                  'SIZE_HIDDEN_LAYERS': 64,
                  'NUM_FILTERS': 25,
                  'NUM_CONV_LAYERS': 3,
                  'OBS_STATE_SHAPE': (y, x)}
}
