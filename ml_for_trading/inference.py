import os
import json
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from data_processor import load_and_fuse_data
from label_generator import generate_labels
from rl_env import TradingEnv

def run_inference():
    # Load the matching hyperparameters that were used to train the saved model
    params_path = os.path.join(os.path.dirname(__file__), 'best_params.json')
    if not os.path.exists(params_path):
        print(f"Hyperparameters file not found at {params_path}. Please run optimizer.py first.")
        return
        
    with open(params_path, 'r') as f:
        best_params = json.load(f)

    print(f"Loading data for inference using parameters: {best_params}")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    df = load_and_fuse_data(data_dir, best_params["norm_window"], best_params["atr_window"])
    df = generate_labels(df, best_params["horizon"], best_params["volatility_multiplier"])
    
    # We create the environment exactly as we did in training
    inference_env_fn = lambda: TradingEnv(df, best_params["seq_length"], best_params["stop_loss_multiplier"])
    inference_env = make_vec_env(inference_env_fn, n_envs=1)
    
    model_path = os.path.join(os.path.dirname(__file__), 'best_trading_model.zip')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run optimizer.py first to train and save the ultimate model.")
        return

    print("Loading specialized model...")
    # Load model enforcing evaluation mode
    model = PPO.load(model_path, env=inference_env, device="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    print("Running inference over the dataset...")
    obs = inference_env.reset()
    done = [False]
    
    actions_taken = []
    rewards_history = []
    
    # Walk through the dataset
    while not done[0]:
        # Standard deterministic prediction (argmax of probabilities) without threshold
        action, _states = model.predict(obs, deterministic=True)
                
        # We step the environment and unpack actual PnL
        obs, reward, done, info = inference_env.step(action)
        actual_pnl = info[0]['actual_pnl'] # Use true PnL tracked in the env's info dict
        
        actions_taken.append(action[0])
        rewards_history.append(actual_pnl)
        
    print("Inference full sweep complete.")
    
    cumulative_pnl = np.cumsum(rewards_history)
    total_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0
    
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = running_max - cumulative_pnl
    max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0
    
    # Track trade executions based on position changes
    mapped_actions = np.array([0 if a == 0 else (1 if a == 1 else -1) for a in actions_taken])
    position_changes = np.diff(mapped_actions, prepend=0)
    total_trades = np.count_nonzero(position_changes)
    
    print("\n" + "="*30)
    print("--- INFERENCE PNL REPORT ---")
    print("="*30)
    print(f"Total Cumulative PnL:  {total_pnl:,.2f} Index Points")
    print(f"Max Drawdown:         -{max_drawdown:,.2f} Index Points")
    print(f"Total Trades Executed: {total_trades}")
    print(f"Average PnL per Trade: {(total_pnl / total_trades) if total_trades > 0 else 0:,.2f} Index Points")
    print("-" * 30)
    print("Distribution of actions taken:")
    print(f"   Hold : {actions_taken.count(0)}")
    print(f"   Buy  : {actions_taken.count(1)}")
    print(f"   Sell : {actions_taken.count(2)}")
    print("="*30 + "\n")

if __name__ == "__main__":
    run_inference()
