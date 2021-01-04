import torch
import os
import random
import copy
import sys
import numpy as np

from rl_networks import MLPNetwork, ConvMLPNetwork
from utilities.Utility_Functions import normalise_rewards, create_actor_distribution
from utilities.rl_utils import setup_logger


class PPOTrainer():
    """PPO Algorithm to play with OpenAI Gym environments"""

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

    def step(self, observation):
        self.timesteps += 1
        state = torch.from_numpy(observation).float()
        action, action_log_prob, _, value = self.policy.act(state)
        self.record_timestep(observation, action, value, action_log_prob)
        return action.item()

    def policy_learn(self):
        for _ in range(self.config['learning_iterations_per_round']):
            advantage, returns_batched = self.calc_advantage()

            states = torch.Tensor(self.states_batched).float()
            actions = torch.Tensor(self.actions_batched).float()

            _, action_log_probs, entropy, value = self.policy.act(
                states, actions)
            ratio = torch.exp(action_log_probs - torch.Tensor(self.log_prob_batched))
            surr1 = ratio * advantage
            surr2 = torch.clamp(
                ratio, 1.0 - self.config['clip_epsilon'], 1.0 + self.config['clip_epsilon']) * advantage

            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns_batched - value).pow(2).mean()
            entropy = entropy.mean()

            self.take_optim_step(value_loss, action_loss, entropy)

    def calc_returns(self):
        returns_batched = []
        for episode_rewards in self.rewards_batched:
            returns = []
            cumulative = 0
            for reward in episode_rewards[::-1]:
                cumulative = reward + self.config['discount_rate'] * cumulative
                returns.append(cumulative)
            returns.reverse()
            returns_batched.append(returns)
        return returns_batched

    def calc_gae(self):
        returns_batched = []
        for episode, episode_rewards in enumerate(self.rewards_batched):
            returns = []
            gae = 0
            values = self.values_batched[episode] + [0]
            for step in reversed(range(len(episode_rewards))):
                delta = episode_rewards[step] + \
                    self.config['discount_rate'] * values[step + 1] - values[step]
                gae = delta + self.config['discount_rate'] * self.config['gae_lambda'] * gae
                returns.insert(0, gae + values[step])
            returns_batched.append(returns)
        return returns_batched

    def calc_advantage(self):
        returns_batched = []
        if self.config['use_gae']:
            returns_batched = self.calc_gae()
        else:
            returns_batched = self.calc_returns()
        advantage = torch.Tensor(returns_batched) - torch.Tensor(self.values_batched)
        advantage = (advantage - advantage.mean(dim=1, keepdim=True)) / \
            (advantage.std(dim=1, keepdim=True) + 1e-5)
        return advantage, torch.Tensor(returns_batched)

    def take_optim_step(self, value_loss, action_loss, dist_entropy):
        self.policy_optim.zero_grad()
        loss = value_loss + action_loss - (dist_entropy * self.config['entropy_coef'])
        print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                       self.config['gradient_clipping_norm'])
        self.policy_optim.step()

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
