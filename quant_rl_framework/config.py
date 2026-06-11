from dataclasses import dataclass

@dataclass(frozen=True)
class TradingConfig:
    INITIAL_BALANCE: float = 100000.0
    TRANSACTION_FEE_PCT: float = 0.0005
    SLIPPAGE_MAX_TICKS: int = 2
    LATENCY_MINUTES: int = 0  # Changed to 0 to avoid complex lookahead issues
    THETA_DECAY_COEFF: float = 0.05 # Increased base theta penalty
    OVERNIGHT_HOLD_PENALTY: float = 15.0 # Penalty for holding overnight
    WINDOW_SIZE: int = 30
    RUIN_THRESHOLD: float = 0.80
    MAX_STEPS_PER_EPISODE: int = 10000 # Randomly sample 1-month slices during training
