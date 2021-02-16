import torch
import os
import random
import numpy as np

from utilities.rl_utils import setup_logger
from rl_networks import MLPNetwork, ConvMLPNetwork


class BaseActorCritic():
    """Base class for actor-critic style algorithms"""

    def __init__(self, config, obs_space, action_space):
        self.config = config
        self.logger = setup_logger(os.getcwd())
        self.obs_space = obs_space
        self.action_space = action_space
        self.set_random_seeds(self.config['seed'])
        self.device = "cuda:0" if self.config['use_GPU'] and torch.cuda.is_available() else "cpu"
        self.episode_number = 1
        self.timesteps = 0
        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.all_values = []
        self.all_log_prob = []
        self.states_batched = []
        self.actions_batched = []
        self.rewards_batched = []
        self.values_batched = []
        self.log_prob_batched = []
        self.average_score_required_to_win = 195.0
        self.init_NN()
        self.reset_game()

    def record_timestep(self, observation, action, value, action_log_prob):
        self.current_episode_state.append(observation)
        self.current_episode_action.append(action)
        self.current_episode_value.append(value)
        self.current_episode_log_prob.append(action_log_prob)

    def receive_rewards(self, reward):
        self.current_episode_reward.append(reward)

    def end_episode(self):
        self.states_batched.append(self.current_episode_state)
        self.actions_batched.append(self.current_episode_action)
        self.rewards_batched.append(self.current_episode_reward)
        self.values_batched.append(self.current_episode_value)
        self.log_prob_batched.append(self.current_episode_log_prob)

        if (self.episode_number % self.config['episodes_per_learning_round'] == 0):
            self.policy_learn()
            # self.update_learning_rate(self.config['policy']['learning_rate'], self.policy_optim)
            self.init_batch_lists()
        self.reset_game()
        self.episode_number += 1
        self.timesteps = 0

    def reset_game(self):
        self.current_episode_state = []
        self.current_episode_action = []
        self.current_episode_reward = []
        self.current_episode_value = []
        self.current_episode_log_prob = []

    def set_random_seeds(self, random_seed=None):
        """Sets all possible random seeds so results can be reproduced"""
        if not random_seed:
            random_seed = np.random.randint(100)
            self.logger.info("Random seed @ {}".format(random_seed))
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)

    def init_batch_lists(self):
        if self.states_batched:
            self.all_states.extend(self.states_batched)
            self.all_actions.extend(self.actions_batched)
            self.all_rewards.extend(self.rewards_batched)
            self.all_values.extend(self.values_batched)
            self.all_log_prob.extend(self.log_prob_batched)

        self.states_batched = []
        self.actions_batched = []
        self.rewards_batched = []
        self.values_batched = []
        self.log_prob_batched = []

    def create_NN(self, NN_type, NN_params):
        NN_dict = {'conv': ConvMLPNetwork,
                   'mlp': MLPNetwork}

        return NN_dict[NN_type](NN_params, self.obs_space, self.action_space)

    def init_NN(self):
        policy_type = self.config['policy']['type']
        params = self.config[policy_type]['params']

        self.policy = self.create_NN(policy_type,
                                     params)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(),
                                             lr=self.config['policy']['learning_rate'], eps=1e-4)

    def step(self):
        raise NotImplementedError

    def policy_learn(self):
        raise NotImplementedError

    def flatten_array(self, array):
        flattened = [time_step for episode in array for time_step in episode]
        return np.vstack(flattened)
