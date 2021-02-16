import torch
import os
import random
import copy
import sys
import numpy as np

from ac_base import BaseActorCritic


class PPOTrainer(BaseActorCritic):
    """PPO Algorithm to play with OpenAI Gym environments"""

    def __init__(self, config, obs_space, action_space):
        super().__init__(config, obs_space, action_space)

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
        loss = (self.config['value_coef'] * value_loss) + action_loss - \
            (dist_entropy * self.config['entropy_coef'])
        print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                       self.config['gradient_clipping_norm'])
        self.policy_optim.step()


class ActorCriticAgent(BaseActorCritic):
    """ Class for vanilla actor-critic """

    def __init__(self, config, obs_space, action_space):
        super().__init__(config, obs_space, action_space)

    def step(self, observation):
        self.timesteps += 1
        state = torch.from_numpy(observation).float()
        action, _, _, value = self.policy.act(state)
        self.record_timestep(observation, action, value, None)
        return action.item()

    def policy_learn(self):
        discounted_rewards = self.calc_discounted_rewards()
        loss = self.calc_loss(discounted_rewards)
        self.take_optim_step(loss)

    def calc_discounted_rewards(self):
        discounted_rewards = []
        for rewards_episode in self.rewards_batched:
            rewards = []
            cumulative = 0
            for reward in rewards_episode[::-1]:
                cumulative = reward + self.config['discount_rate'] * cumulative
                rewards.insert(0, cumulative)
            discounted_rewards.append(rewards)
        return discounted_rewards

    def calc_loss(self, discounted_rewards):
        states = self.flatten_array(self.states_batched)
        states = torch.from_numpy(states).float()
        actions = self.flatten_array(self.actions_batched)
        actions = torch.from_numpy(actions).float()

        _, log_prob, entropy, value = self.policy.act(states, actions)

        rewards = self.flatten_array(discounted_rewards)
        advantage = torch.Tensor(rewards) - value

        value_loss = advantage.pow(2).mean()
        policy_loss = -(log_prob * advantage.detach()).mean()

        loss = self.config['value_coef'] * value_loss + \
            policy_loss - self.config['entropy_coef'] * entropy.mean()
        print("Loss: ", loss.item())
        return loss

    def take_optim_step(self, loss):
        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
