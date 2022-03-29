import logging
import gym
import os
import cv2

from collections import deque

env = gym.make('Breakout-v0', render_mode='human')

obs = env.reset()
print(obs)
print(obs.shape)





