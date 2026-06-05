import os
import logging
import multiprocessing
import dataclasses
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from config import TradingConfig
from data_processor import MarketDataPipeline
from trading_env import OptionsProxyEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_env(csv_path: str, config: TradingConfig):
    def _init():
        pipeline = MarketDataPipeline(csv_path, config)
        df = pipeline.build_features()
        df = pipeline.generate_rolling_scaling(df)
        env = OptionsProxyEnv(df, config)
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
    
    num_envs = multiprocessing.cpu_count()
    env = SubprocVecEnv([make_env(dataset_path, config) for _ in range(num_envs)])
    eval_env = SubprocVecEnv([make_env(dataset_path, config)])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'./models/{run.id}/',
        log_path=f'./logs/{run.id}/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    wandb_callback = WandbCallback(
        model_save_path=f"./models/{run.id}/",
        verbose=2,
    )
    
    callback_list = CallbackList([eval_callback, wandb_callback])
    
    try:
        # We must use RecurrentPPO for MlpLstmPolicy
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
            tensorboard_log=f"runs/{run.id}",
            policy_kwargs=dict(lstm_hidden_size=64, n_lstm_layers=1)
        )
    except ImportError as e:
        logger.error(f"Failed to import sb3_contrib. Ensure it is installed via pip install sb3-contrib. {e}")
        raise

    logger.info("Starting training...")
    model.learn(total_timesteps=100000, callback=callback_list)
    
    model.save("ppo_trading_model")
    run.finish()
    logger.info("Training completed.")

if __name__ == "__main__":
    main()
