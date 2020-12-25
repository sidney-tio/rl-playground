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
        self.states_batched = []
        self.actions_batched = []
        self.rewards_batched = []
        self.average_score_required_to_win = 195.0
        self.init_NN()
        self.reset_game()

    def init_batch_lists(self):
        if self.states_batched:
            self.all_states.extend(self.states_batched)
            self.all_actions.extend(self.actions_batched)
            self.all_rewards.extend(self.rewards_batched)
        self.states_batched = []
        self.actions_batched = []
        self.rewards_batched = []

    def create_NN(self, NN_type, NN_params):
        NN_dict = {'conv': ConvMLPNetwork,
                   'mlp': MLPNetwork}

        return NN_dict[NN_type](NN_params, self.obs_space, self.action_space)

    def init_NN(self):
        policy_type = self.config['policy']['type']
        params = self.config[policy_type]['params']

        self.policy_new = self.create_NN(policy_type,
                                         params)

        self.policy_old = self.create_NN(policy_type,
                                         params)

        self.policy_old.load_state_dict(copy.deepcopy(self.policy_new.state_dict()))
        self.policy_new_optim = torch.optim.Adam(self.policy_new.parameters(),
                                                 lr=self.config['policy']['learning_rate'], eps=1e-4)
        print(self.policy_new)

    def step(self, observation):
        self.timesteps += 1
        state = torch.from_numpy(observation).float()
        actor_output = self.policy_new.forward(state)
        # hardcoded for our own usecase
        action_distribution = create_actor_distribution("DISCRETE", actor_output, self.action_space)
        action = action_distribution.sample().cpu().item()
        self.record_timestep(observation, action)
        return action

    def record_timestep(self, observation, action):
        self.current_episode_state.append(observation)
        self.current_episode_action.append(action)

    def receive_rewards(self, reward):
        self.current_episode_reward.append(reward)

    def policy_learn(self):
        all_discounted_returns = self.calculate_all_discounted_returns()
        if self.config['normalise_rewards']:
            all_discounted_returns = normalise_rewards(all_discounted_returns)

        for _ in range(self.config["learning_iterations_per_round"]):
            all_ratio_of_policy_probabilities = self.calculate_all_ratio_of_policy_probabilities()
            loss = self.calculate_loss([all_ratio_of_policy_probabilities], all_discounted_returns)
            self.take_policy_new_optimisation_step(loss)
        self.init_batch_lists()

    def take_policy_new_optimisation_step(self, loss):
        """Takes an optimisation step for the new policy"""
        self.policy_new_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(),
                                       self.config["gradient_clipping_norm"])
        self.policy_new_optim.step()

    def calculate_all_ratio_of_policy_probabilities(self):
        """For each action calculates the ratio of the probability that the new policy would have picked the action vs.
         the probability the old policy would have picked it. This will then be used to inform the loss"""
        all_states = [state for states in self.states_batched for state in states]
        all_actions = [[action] for actions in self.actions_batched for action in actions]
        all_states = torch.stack([torch.Tensor(states).float().to(self.device)
                                  for states in all_states])

        all_actions = torch.stack([torch.Tensor(actions).float().to(self.device)
                                   for actions in all_actions])
        all_actions = all_actions.view(-1, len(all_states))

        new_policy_distribution_log_prob = self.calculate_log_probability_of_actions(
            self.policy_new, all_states, all_actions)
        old_policy_distribution_log_prob = self.calculate_log_probability_of_actions(
            self.policy_old, all_states, all_actions)
        ratio_of_policy_probabilities = torch.exp(
            new_policy_distribution_log_prob) / (torch.exp(old_policy_distribution_log_prob) + 1e-8)
        return ratio_of_policy_probabilities

    def calculate_log_probability_of_actions(self, policy, states, actions):
        """Calculates the log probability of an action occuring given a policy and starting state"""
        policy_output = policy.forward(states).to(self.device)
        policy_distribution = create_actor_distribution(
            "DISCRETE", policy_output, self.action_space)
        policy_distribution_log_prob = policy_distribution.log_prob(actions)
        return policy_distribution_log_prob

    def calculate_loss(self, all_ratio_of_policy_probabilities, all_discounted_returns):
        """Calculate PPO loss"""
        all_ratio_of_policy_probabilities = torch.squeeze(
            torch.stack(all_ratio_of_policy_probabilities))
        all_ratio_of_policy_probabilities = torch.clamp(input=all_ratio_of_policy_probabilities,
                                                        min=-sys.maxsize,
                                                        max=sys.maxsize)
        all_discounted_returns = torch.tensor(
            all_discounted_returns).to(all_ratio_of_policy_probabilities)
        potential_loss_value_1 = all_discounted_returns * all_ratio_of_policy_probabilities
        potential_loss_value_2 = all_discounted_returns * \
            self.clamp_probability_ratio(all_ratio_of_policy_probabilities)
        loss = torch.min(potential_loss_value_1, potential_loss_value_2)
        loss = -torch.mean(loss)
        self.logger.info(f'Loss: {loss}')
        return loss

    def clamp_probability_ratio(self, value):
        """Clamps a value between a certain range determined by hyperparameter clip epsilon"""
        return torch.clamp(input=value, min=1.0 - self.config["clip_epsilon"],
                           max=1.0 + self.config["clip_epsilon"])

    def calculate_all_discounted_returns(self):
        """Calculates the cumulative discounted return for each episode which we will then use in a learning iteration"""
        all_discounted_returns = []
        for episode in range(len(self.states_batched)):
            discounted_returns = [0]
            for ix in range(len(self.states_batched[episode])):
                return_value = self.rewards_batched[episode][-(ix + 1)] + \
                    self.config['discount_rate']*(discounted_returns[-1])
                discounted_returns.append(return_value)
            discounted_returns = discounted_returns[1:]
            all_discounted_returns.extend(discounted_returns[::-1])
        return all_discounted_returns

    def end_episode(self):
        self.states_batched.append(self.current_episode_state)
        self.actions_batched.append(self.current_episode_action)
        self.rewards_batched.append(self.current_episode_reward)

        if (self.episode_number % self.config['episodes_per_learning_round'] == 0):
            self.policy_learn()
            self.update_learning_rate(self.config['policy']['learning_rate'], self.policy_new_optim)
            self.equalise_policies()
        self.reset_game()
        self.episode_number += 1
        self.timesteps = 0

    def reset_game(self):
        self.current_episode_state = []
        self.current_episode_action = []
        self.current_episode_reward = []

    def equalise_policies(self):
        """Sets the old policy's parameters equal to the new policy's parameters"""
        for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
            old_param.data.copy_(new_param.data)

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

    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if len(self.rewards_batched) > 0:
            last_rolling_score = self.rewards_batched[-1][-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            if random.random() < 0.001:
                self.logger.info("Learning rate {}".format(new_lr))
