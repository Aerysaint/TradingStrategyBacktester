import os
import logging
import multiprocessing
import dataclasses
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from config import TradingConfig
from data_processor import MarketDataPipeline
from trading_env import OptionsProxyEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd

import torch

def make_env(dataset_path: str, config: TradingConfig):
    def _init():
        # Build features INSIDE the spawned process to avoid Windows IPC pipe deadlocks
        pipeline = MarketDataPipeline(dataset_path, config)
        df = pipeline.build_features()
        df = pipeline.generate_rolling_scaling(df)
        
        env = OptionsProxyEnv(df, config)
        env = Monitor(env)
        return env
    return _init

def main():
    config = TradingConfig()
    dataset_path = os.path.join(os.getcwd(), "datasets", "NIFTY 50_minute.csv")
    
    # Initialize Weights & Biases
    run = wandb.init(
        project="momentum-scalping-rl",
        config=dataclasses.asdict(config),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    
    # Cap CPU cores
    num_envs = min(multiprocessing.cpu_count(), 4)
    
    import sys
    if sys.platform == "win32":
        logger.info(f"Windows detected: Vectorizing {num_envs} CPU environments using DummyVecEnv to prevent IPC pipe deadlocks...")
        env = DummyVecEnv([make_env(dataset_path, config) for _ in range(num_envs)])
    else:
        logger.info(f"Vectorizing {num_envs} CPU parallel environments using SubprocVecEnv...")
        env = SubprocVecEnv([make_env(dataset_path, config) for _ in range(num_envs)])
    
    # Eval env
    eval_env = DummyVecEnv([make_env(dataset_path, config)])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/{run.id}/',
        log_path=f'./logs/{run.id}/',
        eval_freq=50000,
        n_eval_episodes=1,
        deterministic=True,
        render=False
    )
    
    wandb_callback = WandbCallback(
        model_save_path=f"./models/{run.id}/",
        verbose=2,
    )
    
    callback_list = CallbackList([eval_callback, wandb_callback])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,
            device=device,
            tensorboard_log=f"runs/{run.id}",
            policy_kwargs=dict(lstm_hidden_size=64, n_lstm_layers=1)
        )
    except ImportError as e:
        logger.error(f"Failed to import sb3_contrib. Ensure it is installed via pip install sb3-contrib. {e}")
        raise

    logger.info("Starting training...")
    model.learn(total_timesteps=3000000, callback=callback_list)
    
    model.save("ppo_trading_model")
    run.finish()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
