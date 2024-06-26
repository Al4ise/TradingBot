import os
import gym_anytrading
import torch
from gym_anytrading.envs import StocksEnv
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

from finta import TA

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import yfinance as yf
from datetime import datetime
import time

def main():
    # Download and Format Data
    ticker = "AAPL"
    df = buildStockData(ticker)
    #print(df.dtypes)
    #print(df.head(10))
    
    #randomSteps(df)

    # train model
    start_time = time.time()
    trainModel(ticker, df, 100000)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # eval
    evaluateModel(ticker, df)

def randomSteps(df):
    env = gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)
    #env.unwrapped.signal_features

    state = env.reset()
    # Test env, random steps
    while True:
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Info: ", info)
            break

    plt.figure(figsize=(15,6))
    plt.cla()
    env.render()
    plt.show()

def buildStockData(ticker):
    csvpath = f'data/{ticker}.csv'
    if not os.path.isfile(csvpath):
        # Get the data for the stock
        print("[*] Downloading Data...")
        data = yf.download(ticker)

        # Save the data to a CSV file
        data.to_csv(f'data/{ticker}.csv')
        print("[*] Downloaded")

    df = formatStockData(csvpath)

    df = applyTechnicals(df)

    return df

def formatStockData(csvpath):
    df = pd.read_csv(csvpath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_values('Date', ascending=True, inplace=True)
    df['Volume'] = df['Volume'].astype(float)
    return df

def applyTechnicals(df):
    #df['SMA'] = TA.SMA(df, 7)
    #df['MACD'] = TA.MACD(df)
    df['RSI'] = TA.RSI(df)
    #df['RSI'] = TA.

    #df['MACD_signal'], df['MACD_histogram'] = TA.MACD(df)
    #df['MACD_signal'] = df['MACD_signal'].astype(float)
    #df['MACD_histogram'] = df['MACD_histogram'].astype(float)

    #df['VWAP'] = TA.VWAP(df)

    df.fillna(0, inplace=True)

    return df

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'High', 'Volume', 'Open', 'Close', 'RSI']].to_numpy()[start:end]
    return prices, signal_features

def trainModel(ticker, df, timesteps):
    env_maker = lambda: customEnv(df=df, frame_bound=(5, 10000), window_size=5)
    env = make_vec_env(env_maker, vec_env_cls=SubprocVecEnv, n_envs=20)

    #model = A2C('MlpPolicy', env, device='cuda', verbose=1)
    model = A2C('MlpPolicy', env, device='cpu', verbose=1)
    #model = A2C('MlpPolicy', env, device='cpu', verbose=1, policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))

    model.learn(total_timesteps=timesteps)
    model.save(f'models/{ticker}')

def evaluateModel(ticker, df):
    env_maker = lambda: customEnv(df=df, frame_bound=(10000, 11000), window_size=5, render_mode='human')
    env = make_vec_env(env_maker, vec_env_cls=SubprocVecEnv)
    model = A2C.load(f'models/{ticker}')

    obs = env.reset()
    while True:
        obs = np.array(obs)  # Convert obs to a NumPy array
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("Info: ", info)
            break

    plt.figure(figsize=(15,6))
    plt.cla()
    env.render()
    plt.show()

class customEnv(StocksEnv):
    _process_data = add_signals

if __name__ == '__main__':
    main()

