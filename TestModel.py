import os
import gym_anytrading
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

model = A2C.load("AAPL_Test")
# Load the test data
test_data = pd.read_csv("data/AAPL.csv")

# Preprocess the test data
frame1 = 500
#frame2 = len(test_data)
frame2 = 550
test_env = lambda: gym.make("stocks-v0", df=test_data, frame_bound=(frame1, frame2), window_size=500, render_mode = 'human')

# Wrap the test environment in a vectorized environment
test_env = make_vec_env(test_env)

# Evaluate the model on the test environment
obs = test_env.reset()
done = False
while True:
    action, _ = model.predict(obs)
    n, _, done, info = test_env.step(action)
    if done:
        print("Info: ", info)
        break

# Visualize the test results
plt.figure(figsize=(15,6))
plt.cla()
test_env.render()
plt.show()
