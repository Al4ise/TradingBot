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


from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.main import check_and_make_directories
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR

check_and_make_directories([TRAINED_MODEL_DIR])


from finta import TA

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import yfinance as yf
from datetime import datetime
import time
import itertools


def main():
    # Download and Format Data
    name = "Model"
    symbols = [
        'MSTR',
        'AAPL',
        'NVDA',
        'MSFT',
        'SMCI'
        # 'ARM'
    ]

    # Save train and test data to different CSV files
    train_path, test_path = buildStockData(symbols)

    train = pd.read_csv(train_path)
    train = train.set_index(train.columns[0])
    train.index.names = ['']

    test = pd.read_csv(test_path)
    test = test.set_index(test.columns[0])
    test.index.names = ['']

    # train model
    start_time = time.time()
    trainModel(train)
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # eval
    evaluateModel(name, df)

def makeGym(train):
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    return env_train

def buildStockData(symbols):
    if not os.path.exists('data'):
        os.makedirs('data')
        
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        TRAIN_START_DATE = '2009-01-01'
        TRAIN_END_DATE = '2023-05-01'

        TEST_START_DATE = '2023-05-02'
        TEST_END_DATE = '2024-05-01'

        df_raw = YahooDownloader(start_date=TRAIN_START_DATE, end_date=TEST_END_DATE, ticker_list=symbols).fetch_data()

        fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=INDICATORS,
                            use_turbulence=True,
                            use_vix=True,
                            user_defined_feature=False)


        processed = fe.preprocess_data(df_raw)

        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
        combination = list(itertools.product(list_date,list_ticker))

        processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
        processed_full = processed_full[processed_full['date'].isin(processed['date'])]
        processed_full = processed_full.sort_values(['date','tic'])

        processed_full = processed_full.fillna(0)

        train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
        test = data_split(processed_full, TEST_START_DATE, TEST_END_DATE)

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

    return train_path, test_path


def trainModel(train):
    env_train = makeGym(train)

    agent = DRLAgent(env = env_train)

    # Set the corresponding values to 'True' for the algorithms that you want to use
    if_using_a2c = True
    if_using_ddpg = False
    if_using_ppo = False
    if_using_td3 = False
    if_using_sac = False

    model_a2c = agent.get_model("a2c")
    model_ppo = agent.get_model('ppo')

    if if_using_a2c:
        # set up logger
        tmp_path = RESULTS_DIR + '/a2c'
        new_logger_a2c = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_a2c.set_logger(new_logger_a2c)

    if if_using_ppo:
        # set up logger
        tmp_path = RESULTS_DIR + '/ppo'
        new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
        # Set new logger
        model_ppo.set_logger(new_logger_ppo)

    trained_a2c = agent.train_model(model=model_a2c,tb_log_name='a2c',total_timesteps=50000) if if_using_a2c else None

    trained_ppo = agent.train_model(model=model_ppo,tb_log_name='ppo',total_timesteps=50000) if if_using_ppo else None
       
    trained_a2c.save(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None # type: ignore
    trained_ppo.save(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None # type: ignore
       

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

if __name__ == '__main__':
    main()

