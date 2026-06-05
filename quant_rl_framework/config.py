from dataclasses import dataclass

@dataclass(frozen=True)
class TradingConfig:
    INITIAL_BALANCE: float = 100000.0
    TRANSACTION_FEE_PCT: float = 0.0005
    SLIPPAGE_MAX_TICKS: int = 2
    LATENCY_MINUTES: int = 1
    THETA_DECAY_COEFF: float = 0.01
    WINDOW_SIZE: int = 30
    RUIN_THRESHOLD: float = 0.80
