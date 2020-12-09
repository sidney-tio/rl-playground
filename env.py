import gym

env = gym.make('CartPole-v0')

env.reset()

for _ in range(100):
    env.render()
    print(env.step(env.action_space.sample()))

env.close()
