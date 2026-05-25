from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Union

from .data import init_db, get_ohlcv_data
from .backtest import run_backtest

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up, initializing database...")
    init_db()
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IndicatorDef(BaseModel):
    id: str
    name: str 
    params: Dict[str, Any]

class ConditionDef(BaseModel):
    left: str 
    operator: str 
    right: Union[str, float] 

class StrategyRule(BaseModel):
    action: str 
    conditions: List[ConditionDef]
    timeframe: str = None

class OptionsConfig(BaseModel):
    enabled: bool = False
    dte: int = 7
    iv: float = 0.15
    risk_free_rate: float = 0.05

class BacktestRequest(BaseModel):
    dataset: str = "1minute"
    start_date: str = None
    end_date: str = None
    indicators: List[IndicatorDef] = []
    rules: List[StrategyRule] = []
    options: OptionsConfig = OptionsConfig()


@app.get("/api/data")
def get_data(dataset: str = "1minute", start: str = None, end: str = None, limit: int = 5000):
    df = get_ohlcv_data(dataset, start, end, limit)
    return df.to_dict(orient="records")

@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    df = get_ohlcv_data(req.dataset, req.start_date, req.end_date, limit=200000)
    options_dict = req.options.dict() if hasattr(req.options, 'dict') else {}
    results = run_backtest(df, req.indicators, req.rules, req.dataset, options_dict)
    return results

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
