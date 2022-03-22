import gym

env = gym.make('PongNoFrameskip-v4')
env.reset()

for _ in range(30):
    obs, rew, done, info = env.step(0)
    print(info)