import optuna
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
from typing import Callable

from data_processor import load_and_fuse_data
from label_generator import generate_labels
from rl_env import TradingEnv

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def optimize_agent(trial):
    # 1. Sample hyperparameters
    seq_length = trial.suggest_int("seq_length", 10, 60)
    norm_window = trial.suggest_int("norm_window", 100, 1000)
    atr_window = trial.suggest_int("atr_window", 5, 30)
    volatility_multiplier = trial.suggest_float("volatility_multiplier", 0.5, 3.0)
    stop_loss_multiplier = trial.suggest_float("stop_loss_multiplier", 0.5, 3.0)
    gamma = trial.suggest_float("gamma", 0.90, 0.999) # Discount factor
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    horizon = trial.suggest_int("horizon", 5, 20)
    
    # data_dir points to datasets relative to current folder
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    
    # 2. Run data processor and label generator
    try:
        df = load_and_fuse_data(data_dir, norm_window, atr_window)
        df = generate_labels(df, horizon, volatility_multiplier)
    except Exception as e:
        return -np.inf

    if len(df) < 1000:
        return -np.inf

    # 3. Time Series Walk-Forward Splitter
    tscv = TimeSeriesSplit(n_splits=3)
    val_sharpes = []
    
    for train_index, val_index in tscv.split(df):
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        
        if len(train_df) < seq_length or len(val_df) < seq_length:
            continue
            
        # 4. Train RL agent (PPO) 
        # Using MlpPolicy but using flattened seq_length * features input natively from TradingEnv
        train_env_fn = lambda: TradingEnv(train_df, seq_length, stop_loss_multiplier, total_training_steps=10000)
        train_env = make_vec_env(train_env_fn, n_envs=1)
        
        model = PPO(
            "MlpPolicy", 
            train_env, 
            learning_rate=linear_schedule(3e-4), 
            ent_coef=0.05,
            gamma=gamma,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
            verbose=0,
            seed=42,
            device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        
        model.learn(total_timesteps=10000) # Fast train for tuning
        
        # 5. Evaluate on Validation folds based on Sharpe Ratio
        val_env = TradingEnv(val_df, seq_length, stop_loss_multiplier)
        obs, _ = val_env.reset()
        done = False
        rewards = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = val_env.step(action)
            rewards.append(reward)
            
        if len(rewards) > 0 and np.std(rewards) > 0:
            # Approx daily sharpe assuming 5m freq (75 periods per day avg for India equities)
            sharpe = np.sqrt(75) * np.mean(rewards) / np.std(rewards)
        else:
            sharpe = -np.inf
            
        val_sharpes.append(sharpe)
        
    if not val_sharpes:
        return -np.inf
        
    return np.mean(val_sharpes)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_agent, n_trials=5) # Can be increased by user
    print("Best hyperparameters:", study.best_params)
    
    # Train final model on the full dataset using the best hyperparameters
    print("\nTraining final model with best hyperparameters on the entire dataset...")
    best_params = study.best_params
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    df = load_and_fuse_data(data_dir, best_params["norm_window"], best_params["atr_window"])
    df = generate_labels(df, best_params["horizon"], best_params["volatility_multiplier"])
    
    final_env_fn = lambda: TradingEnv(df, best_params["seq_length"], best_params["stop_loss_multiplier"], total_training_steps=50000)
    final_env = make_vec_env(final_env_fn, n_envs=1)
    
    final_model = PPO(
        "MlpPolicy", 
        final_env, 
        learning_rate=linear_schedule(3e-4),
        ent_coef=0.05,
        gamma=best_params["gamma"],
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
        verbose=1,
        seed=42,
        device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    
    # Train for more timesteps to ensure convergence on the full dataset
    final_model.learn(total_timesteps=50000)
    
    # Save the model
    model_path = os.path.join(os.path.dirname(__file__), 'best_trading_model')
    final_model.save(model_path)
    
    # Save the best parameters to a JSON file so inference can use the exact matching shapes
    import json
    params_path = os.path.join(os.path.dirname(__file__), 'best_params.json')
    with open(params_path, 'w') as f:
        json.dump(best_params, f)
        
    print(f"\nModel saved successfully to: {model_path}.zip")

