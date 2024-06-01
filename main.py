import os
import gym_anytrading
from gym_anytrading.envs import StocksEnv
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C

from finta import TA

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import yfinance as yf
from datetime import datetime

def main():
    # Download and Format Data
    ticker = "AAPL"
    df = buildStockData(ticker)
    #print(df.dtypes)
    #print(df.head(10))
    
    #randomSteps(df)

    # train model
    trainModel(ticker, df, 2000000)

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
    df['SMA'] = TA.SMA(df, 7)
    #df['MACD'] = TA.MACD(df)
    df['RSI'] = TA.RSI(df)
    #df['VWAP'] = TA.VWAP(df)

    df.fillna(0, inplace=True)

    return df

def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'High', 'Open', 'Close', 'Volume', 'SMA', 'RSI']].to_numpy()[start:end]
    return prices, signal_features

def trainModel(ticker, df, timesteps):
    env_maker = lambda: customEnv(df=df, frame_bound=(6000, 10000), window_size=10)
    env = make_vec_env(env_maker)

    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps, )
    model.save(f'models/{ticker}')

def evaluateModel(ticker, df):
    env_maker = lambda: customEnv(df=df, frame_bound=(10000, 11000), window_size=10, render_mode='human')
    env = make_vec_env(env_maker)
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

