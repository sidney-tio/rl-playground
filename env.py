import gym
import yaml
import numpy as np
from rl_trainer import PPOTrainer


def load_configs():
    with open('configs.yml') as file:
        configs = yaml.load(file)
    return configs


def init_env(env_name):
    env = gym.make(env_name)
    n_observation_space = env.reset().shape
    n_action_space = env.action_space.n
    return env, n_observation_space, n_action_space


def check_solved(total_reward, avg_reward):
    if len(avg_reward) < 100:
        avg_reward.append(np.sum(total_reward))
        print(np.mean(avg_reward))
    else:
        avg_reward.pop(0)
        avg_reward.append(np.sum(total_reward))
        print(np.mean(avg_reward))


def main():
    configs = load_configs()
    env, n_observation_space, n_action_space = init_env(configs['env'])
    agent = PPOTrainer(configs, n_observation_space, n_action_space)

    avg_reward = []
    for episode in range(configs['num_episodes']):
        print(f"Starting episode {episode}")
        observation = env.reset()
        total_reward = []
        for t in range(configs['timesteps']):
            action = agent.step(np.array([observation]))
            observation, reward, done, info = env.step(action)
            agent.receive_rewards(reward)
            total_reward.append(reward)
            if done:
                print(f"Episode finished after {t+1} timesteps")
                agent.end_episode()
                check_solved(total_reward, avg_reward)
                break

    env.close()


if __name__ == "__main__":
    main()
