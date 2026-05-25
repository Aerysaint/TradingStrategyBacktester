'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { ChartRef } from '@/components/Chart';
import dynamic from 'next/dynamic';
import { Play, Settings, RefreshCw, BarChart2, Save, FolderOpen } from 'lucide-react';
import StrategyBuilder from '@/components/StrategyBuilder';

const LightweightChart = dynamic(() => import('@/components/Chart').then(mod => mod.LightweightChart), {
  ssr: false
});

const DEFAULT_STRATEGY = {
  indicators: [
    { id: "fast_ma", name: "sma", params: { length: 10 } },
    { id: "slow_ma", name: "sma", params: { length: 50 } }
  ],
  rules: [
    {
      action: "buy",
      conditions: [{ left: "fast_ma", operator: "crossover", right: "slow_ma" }]
    },
    {
      action: "sell",
      conditions: [{ left: "fast_ma", operator: "crossunder", right: "slow_ma" }]
    }
  ]
};

export default function Home() {
  const [data, setData] = useState<any[]>([]);
  const [markers, setMarkers] = useState<any[]>([]);
  const [indicatorSeries, setIndicatorSeries] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState({
    dataset: '1minute',
    start_date: '2015-01-09',
    end_date: '2015-01-30'
  });
  const [strategy, setStrategy] = useState<any>(DEFAULT_STRATEGY);
  const [savedStrategies, setSavedStrategies] = useState<Record<string, any>>({});
  const [selectedPreset, setSelectedPreset] = useState<string>('');

  const chartRef = useRef<ChartRef>(null);

  useEffect(() => {
    const saved = localStorage.getItem('algo_strategies');
    if (saved) {
      setSavedStrategies(JSON.parse(saved));
    }
  }, []);

  const handleSaveStrategy = () => {
    const name = prompt("Enter strategy name:");
    if (name) {
      const newSaved = { ...savedStrategies, [name]: strategy };
      setSavedStrategies(newSaved);
      localStorage.setItem('algo_strategies', JSON.stringify(newSaved));
      setSelectedPreset(name);
    }
  };

  const handleLoadStrategy = (name: string) => {
    setSelectedPreset(name);
    if (name && savedStrategies[name]) {
      setStrategy(savedStrategies[name]);
    }
  };

  useEffect(() => {
    fetchData();
  }, [params.dataset]);

  const fetchData = async () => {
    setLoading(true);
    // Clear out old indicators and markers so they don't break the new timeframe boundary
    setMarkers([]);
    setIndicatorSeries([]);
    setMetrics(null);
    try {
      const query = new URLSearchParams({
        dataset: params.dataset,
        start: params.start_date,
        end: params.end_date,
        limit: '200000'
      });
      const res = await fetch(`http://localhost:8000/api/data?${query}`);
      const rawData = await res.json();
      
      const formattedData = rawData.map((d: any) => ({
        time: (new Date(d.date + "Z")).getTime() / 1000, 
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
        volume: d.volume
      }));
      setData(formattedData);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const handleBacktest = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset: params.dataset,
          start_date: params.start_date,
          end_date: params.end_date,
          indicators: strategy.indicators || [],
          rules: strategy.rules || []
        })
      });
      
      const result = await res.json();
      
      if(result.error) {
         alert("Error: " + result.error);
         setLoading(false);
         return;
      }
      
      setMetrics(result.metrics);
      setIndicatorSeries(result.indicators || []);

      const newMarkers = result.trades.map((t: any) => ({
        time: (new Date(t.time + "Z")).getTime() / 1000,
        position: t.action === 'buy' ? 'belowBar' : 'aboveBar',
        color: t.action === 'buy' ? '#26a69a' : '#ef5350',
        shape: t.action === 'buy' ? 'arrowUp' : 'arrowDown',
        text: t.action === 'buy' ? `Buy @ ${t.price.toFixed(2)}` : `Sell @ ${t.price.toFixed(2)}`
      }));
      
      setMarkers(newMarkers);
    } catch (e) {
      console.error(e);
    }

    setLoading(false);
  };

  return (
    <div className="flex h-screen bg-[#131722] text-[#d1d4dc] overflow-hidden font-sans">
      <aside suppressHydrationWarning className="w-[400px] bg-[#1e222d] border-r border-[#2b2b2b] flex flex-col z-10 transition-all">
        <div className="p-4 border-b border-[#2b2b2b] flex items-center justify-between">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <BarChart2 className="w-6 h-6 text-blue-500" />
            <span className="text-white">AlgoTerminal</span>
          </h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-6">
          <section>
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Settings className="w-4 h-4" /> Parameters
            </h2>
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-xs text-gray-400">Dataset (Timeframe)</label>
                <select
                  value={params.dataset}
                  onChange={e => setParams({...params, dataset: e.target.value})}
                  className="w-full bg-[#131722] border border-[#2b2b2b] rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 transition-colors text-white"
                >
                  <option value="1minute">1 Minute</option>
                  <option value="5minute">5 Minutes</option>
                  <option value="15minute">15 Minutes</option>
                  <option value="60minute">60 Minutes</option>
                  <option value="day">Daily</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-xs text-gray-400">Start Date</label>
                  <input 
                    type="date" 
                    value={params.start_date.split('T')[0]}
                    onChange={e => setParams({...params, start_date: e.target.value})}
                    className="w-full bg-[#131722] border border-[#2b2b2b] rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 transition-colors [color-scheme:dark]"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-xs text-gray-400">End Date</label>
                  <input 
                    type="date" 
                    value={params.end_date.split('T')[0]}
                    onChange={e => setParams({...params, end_date: e.target.value})}
                    className="w-full bg-[#131722] border border-[#2b2b2b] rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 transition-colors [color-scheme:dark]"
                  />
                </div>
              </div>
            </div>
          </section>

          <section>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                Strategy Presets
              </h2>
            </div>
            <div className="flex gap-2 mb-4">
              <select 
                value={selectedPreset}
                onChange={e => handleLoadStrategy(e.target.value)}
                className="flex-1 bg-[#131722] border border-[#2b2b2b] rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 transition-colors text-white"
              >
                <option value="">-- Load Strategy --</option>
                {Object.keys(savedStrategies).map(name => (
                   <option key={name} value={name}>{name}</option>
                ))}
              </select>
              <button 
                onClick={handleSaveStrategy}
                className="bg-[#1e222d] border border-[#2b2b2b] hover:border-blue-500 text-gray-400 hover:text-blue-500 px-3 py-2 rounded flex items-center justify-center transition-colors"
                title="Save Current Strategy"
              >
                <Save className="w-4 h-4" />
              </button>
            </div>
          </section>

          <StrategyBuilder strategy={strategy} onChange={setStrategy} />

          <button 
            onClick={handleBacktest}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
            {loading ? 'Running...' : 'Run Simulation'}
          </button>

          {metrics && (
            <section className="bg-[#131722] border border-[#2b2b2b] rounded-lg p-4">
              <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">Performance</h2>
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="bg-[#1e222d] p-2 rounded">
                  <div className="text-[10px] text-gray-400 uppercase">Net PnL</div>
                  <div className={`text-sm font-bold ${metrics.net_pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {metrics.net_pnl >= 0 ? '+' : ''}₹{metrics.net_pnl.toFixed(2)}
                  </div>
                </div>
                <div className="bg-[#1e222d] p-2 rounded">
                  <div className="text-[10px] text-gray-400 uppercase">Win Rate</div>
                  <div className="text-sm font-bold text-white">{metrics.win_rate.toFixed(1)}%</div>
                </div>
                <div className="bg-[#1e222d] p-2 rounded">
                  <div className="text-[10px] text-gray-400 uppercase">Trades</div>
                  <div className="text-sm font-bold text-white">{metrics.total_trades}</div>
                </div>
                <div className="bg-[#1e222d] p-2 rounded">
                  <div className="text-[10px] text-gray-400 uppercase">Max Drawdown</div>
                  <div className="text-sm font-bold text-red-400">{metrics.max_drawdown_pct.toFixed(2)}%</div>
                </div>
              </div>
            </section>
          )}
        </div>
      </aside>

      <main className="flex-1 flex flex-col relative w-full h-full p-4">
        <div className="flex-1 bg-[#1e222d] border border-[#2b2b2b] rounded-lg overflow-hidden shadow-xl">
           <LightweightChart 
              ref={chartRef} 
              data={data} 
              markers={markers}
              indicators={indicatorSeries}
              indicatorDefs={strategy.indicators}
              colors={{
                backgroundColor: '#1e222d',
                textColor: '#d1d4dc',
              }}
            />
        </div>
      </main>
    </div>
  );
}
