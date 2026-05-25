import React, { useState, useEffect } from 'react';
import { Plus, Trash2, ChevronDown, ChevronUp } from 'lucide-react';

const INDICATOR_CATALOG = [
  { name: 'sma', label: 'Simple Moving Average (SMA)', params: { length: 10 } },
  { name: 'ema', label: 'Exponential Moving Average (EMA)', params: { length: 10 } },
  { name: 'rsi', label: 'Relative Strength Index (RSI)', params: { length: 14 } },
  { name: 'macd', label: 'MACD', params: { fast: 12, slow: 26, signal: 9 }, subProps: ['macd', 'macdh', 'macds'] },
  { name: 'stoch', label: 'Stochastic Oscillator', params: { k: 14, d: 3, smooth_k: 3 }, subProps: ['stochk', 'stochd'] },
  { name: 'bbands', label: 'Bollinger Bands', params: { length: 20, std: 2 }, subProps: ['bbl', 'bbm', 'bbu', 'bbb', 'bbp'] },
  { name: 'vwap', label: 'VWAP', params: {} },
];

const OPERATORS = [
  { value: '>', label: 'Greater Than (>)' },
  { value: '<', label: 'Less Than (<)' },
  { value: '>=', label: 'Greater or Equal (>=)' },
  { value: '<=', label: 'Less or Equal (<=)' },
  { value: '==', label: 'Equals (==)' },
  { value: 'crossover', label: 'Crosses Over' },
  { value: 'crossunder', label: 'Crosses Under' },
];

export default function StrategyBuilder({ strategy, onChange }: { strategy: any, onChange: (s: any) => void }) {
  const [expanded, setExpanded] = useState({ indicators: true, buyRules: true, sellRules: true, execution: true });
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  const addIndicator = () => {
    const newInd = { id: `ind_${Date.now()}`, name: 'sma', params: { length: 10 } };
    onChange({ ...strategy, indicators: [...(strategy.indicators || []), newInd] });
  };

  const updateIndicator = (index: number, updates: any) => {
    const newInds = [...strategy.indicators];
    if (updates.name && updates.name !== newInds[index].name) {
       const template = INDICATOR_CATALOG.find(i => i.name === updates.name);
       newInds[index] = { ...newInds[index], ...updates, params: template?.params || {} };
    } else {
       newInds[index] = { ...newInds[index], ...updates };
    }
    onChange({ ...strategy, indicators: newInds });
  };

  const removeIndicator = (index: number) => {
    const newInds = [...strategy.indicators];
    newInds.splice(index, 1);
    onChange({ ...strategy, indicators: newInds });
  };

  const getActionRules = (action: string) => {
    return (strategy.rules || []).find((r: any) => r.action === action)?.conditions || [];
  };

  const updateRules = (action: string, conditions: any[], timeframe?: string) => {
    const rules = [...(strategy.rules || [])];
    const idx = rules.findIndex((r: any) => r.action === action);
    if (idx >= 0) {
      if (timeframe !== undefined) {
         rules[idx] = { ...rules[idx], conditions, timeframe };
      } else {
         rules[idx] = { ...rules[idx], conditions };
      }
    } else {
      rules.push({ action, conditions, timeframe: timeframe || '' });
    }
    onChange({ ...strategy, rules });
  };

  const updateRuleTimeframe = (action: string, timeframe: string) => {
     const conditions = getActionRules(action);
     updateRules(action, conditions, timeframe);
  };

  const addCondition = (action: string) => {
    const conditions = [...getActionRules(action)];
    conditions.push({ left: 'close', operator: '>', right: '0' });
    updateRules(action, conditions);
  };

  const updateCondition = (action: string, index: number, updates: any) => {
    const conditions = [...getActionRules(action)];
    conditions[index] = { ...conditions[index], ...updates };
    updateRules(action, conditions);
  };

  const removeCondition = (action: string, index: number) => {
    const conditions = [...getActionRules(action)];
    conditions.splice(index, 1);
    updateRules(action, conditions);
  };

  const updateOptions = (updates: any) => {
    onChange({ ...strategy, options: { ...(strategy.options || {}), ...updates } });
  };

  // Dynamically compute list of options for conditions datalist
  let indicatorOptions = ['close', 'open', 'high', 'low', 'volume'];
  (strategy.indicators || []).forEach((ind: any) => {
    const cat = INDICATOR_CATALOG.find(c => c.name === ind.name);
    if (cat?.subProps) {
      cat.subProps.forEach(sp => indicatorOptions.push(`${ind.id}.${sp}`));
    } else {
      indicatorOptions.push(ind.id);
    }
  });

  if (!mounted) return null;

  return (
    <div className="space-y-4 text-xs" suppressHydrationWarning>
      {/* INDICATORS SECTION */}
      <div className="bg-[#1e222d] border border-[#2b2b2b] rounded-lg overflow-hidden">
        <div 
          className="p-3 bg-[#131722] border-b border-[#2b2b2b] flex justify-between items-center cursor-pointer select-none"
          onClick={() => setExpanded(prev => ({...prev, indicators: !prev.indicators}))}
        >
          <span className="font-semibold text-gray-300">1. Define Indicators</span>
          {expanded.indicators ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
        {expanded.indicators && (
          <div className="p-3 space-y-3">
            {(strategy.indicators || []).map((ind: any, i: number) => (
              <div key={i} className="bg-[#131722] p-2 rounded border border-[#2b2b2b] space-y-2">
                <div className="flex flex-wrap justify-between items-center gap-2">
                  <input 
                    type="text" 
                    value={ind.id} 
                    onChange={e => updateIndicator(i, { id: e.target.value })}
                    className="w-20 bg-transparent border-b border-gray-600 focus:border-blue-500 outline-none px-1"
                    placeholder="ID"
                  />
                  <select 
                    value={ind.name}
                    onChange={e => updateIndicator(i, { name: e.target.value })}
                    className="flex-1 min-w-[120px] bg-[#1e222d] border border-[#2b2b2b] rounded px-2 py-1 outline-none text-white focus:border-blue-500 truncate"
                  >
                    {INDICATOR_CATALOG.map(cat => <option key={cat.name} value={cat.name}>{cat.label}</option>)}
                  </select>
                  <button onClick={() => removeIndicator(i)} className="text-red-500 hover:text-red-400 p-1 flex-shrink-0"><Trash2 size={14} /></button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {Object.keys(ind.params).map(pKey => (
                    <div key={pKey} className="flex items-center gap-1 bg-[#1e222d] px-2 py-1 rounded border border-[#2b2b2b]">
                      <span className="text-gray-500">{pKey}:</span>
                      <input 
                        type="number" 
                        value={ind.params[pKey]}
                        onChange={e => updateIndicator(i, { params: { ...ind.params, [pKey]: Number(e.target.value) } })}
                        className="w-12 bg-transparent outline-none text-white font-mono"
                      />
                    </div>
                  ))}
                </div>
              </div>
            ))}
            <button onClick={addIndicator} className="w-full flex items-center justify-center gap-1 py-2 border border-dashed border-[#2b2b2b] rounded text-gray-400 hover:text-white hover:border-blue-500 transition-colors">
              <Plus size={14} /> Add Indicator
            </button>
          </div>
        )}
      </div>

      {/* BUY RULES SECTION */}
      <div className="bg-[#1e222d] border border-[#2b2b2b] rounded-lg overflow-hidden">
        <div 
          className="p-3 bg-[#131722] border-b border-[#2b2b2b] flex justify-between items-center cursor-pointer select-none"
          onClick={() => setExpanded(prev => ({...prev, buyRules: !prev.buyRules}))}
        >
          <div className="flex items-center gap-3">
             <span className="font-semibold text-green-500">2. Buy Conditions</span>
             <select
               value={(strategy.rules || []).find((r: any) => r.action === 'buy')?.timeframe || ''}
               onChange={e => updateRuleTimeframe('buy', e.target.value)}
               onClick={e => e.stopPropagation()}
               className="bg-[#1e222d] border border-[#2b2b2b] rounded px-2 py-1 outline-none text-gray-300 focus:border-green-500 text-[10px]"
             >
               <option value="">Base TF</option>
               <option value="1minute">1m</option>
               <option value="5minute">5m</option>
               <option value="15minute">15m</option>
               <option value="60minute">1h</option>
               <option value="day">1D</option>
             </select>
          </div>
          {expanded.buyRules ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
        {expanded.buyRules && (
          <div className="p-3 space-y-3">
            {getActionRules('buy').map((cond: any, i: number) => (
              <div key={i} className="flex flex-col gap-2 bg-[#131722] p-2 rounded border border-[#2b2b2b]">
                <div className="flex items-center gap-2">
                  <input 
                    list="indicatorsList"
                    value={cond.left}
                    onChange={e => updateCondition('buy', i, { left: e.target.value })}
                    className="w-1/3 min-w-0 bg-[#1e222d] border border-[#2b2b2b] text-white rounded px-2 py-1 outline-none focus:border-green-500"
                    placeholder="Value 1"
                  />
                  <select 
                    value={cond.operator}
                    onChange={e => updateCondition('buy', i, { operator: e.target.value })}
                    className="flex-1 bg-[#1e222d] border border-[#2b2b2b] text-white rounded px-1 py-1 outline-none focus:border-green-500"
                  >
                    {OPERATORS.map(op => <option key={op.value} value={op.value}>{op.label}</option>)}
                  </select>
                  <button onClick={() => removeCondition('buy', i)} className="text-red-500 hover:text-red-400 p-1"><Trash2 size={14} /></button>
                </div>
                <input 
                  list="indicatorsList"
                  value={cond.right}
                  onChange={e => updateCondition('buy', i, { right: e.target.value })}
                  className="w-full bg-[#1e222d] border border-[#2b2b2b] text-white rounded px-2 py-1 outline-none focus:border-green-500"
                  placeholder="Value 2"
                />
              </div>
            ))}
            <button onClick={() => addCondition('buy')} className="w-full flex items-center justify-center gap-1 py-2 border border-dashed border-[#2b2b2b] rounded text-green-500 hover:text-green-400 hover:border-green-500 transition-colors">
              <Plus size={14} /> Add Buy Rule
            </button>
          </div>
        )}
      </div>

      {/* SELL RULES SECTION */}
      <div className="bg-[#1e222d] border border-[#2b2b2b] rounded-lg overflow-hidden">
        <div 
          className="p-3 bg-[#131722] border-b border-[#2b2b2b] flex justify-between items-center cursor-pointer select-none"
          onClick={() => setExpanded(prev => ({...prev, sellRules: !prev.sellRules}))}
        >
          <div className="flex items-center gap-3">
             <span className="font-semibold text-red-500">3. Sell Conditions</span>
             <select
               value={(strategy.rules || []).find((r: any) => r.action === 'sell')?.timeframe || ''}
               onChange={e => updateRuleTimeframe('sell', e.target.value)}
               onClick={e => e.stopPropagation()}
               className="bg-[#1e222d] border border-[#2b2b2b] rounded px-2 py-1 outline-none text-gray-300 focus:border-red-500 text-[10px]"
             >
               <option value="">Base TF</option>
               <option value="1minute">1m</option>
               <option value="5minute">5m</option>
               <option value="15minute">15m</option>
               <option value="60minute">1h</option>
               <option value="day">1D</option>
             </select>
          </div>
          {expanded.sellRules ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
        {expanded.sellRules && (
          <div className="p-3 space-y-3">
            {getActionRules('sell').map((cond: any, i: number) => (
              <div key={i} className="flex flex-col gap-2 bg-[#131722] p-2 rounded border border-[#2b2b2b]">
                <div className="flex items-center gap-2">
                  <input 
                    list="indicatorsList"
                    value={cond.left}
                    onChange={e => updateCondition('sell', i, { left: e.target.value })}
                    className="w-1/3 min-w-0 bg-[#1e222d] border border-[#2b2b2b] text-white rounded px-2 py-1 outline-none focus:border-red-500"
                    placeholder="Value 1"
                  />
                  <select 
                    value={cond.operator}
                    onChange={e => updateCondition('sell', i, { operator: e.target.value })}
                    className="flex-1 bg-[#1e222d] border border-[#2b2b2b] text-white rounded px-1 py-1 outline-none focus:border-red-500"
                  >
                    {OPERATORS.map(op => <option key={op.value} value={op.value}>{op.label}</option>)}
                  </select>
                  <button onClick={() => removeCondition('sell', i)} className="text-red-500 hover:text-red-400 p-1"><Trash2 size={14} /></button>
                </div>
                <input 
                  list="indicatorsList"
                  value={cond.right}
                  onChange={e => updateCondition('sell', i, { right: e.target.value })}
                  className="w-full bg-[#1e222d] border border-[#2b2b2b] text-white rounded px-2 py-1 outline-none focus:border-red-500"
                  placeholder="Value 2"
                />
              </div>
            ))}
            <button onClick={() => addCondition('sell')} className="w-full flex items-center justify-center gap-1 py-2 border border-dashed border-[#2b2b2b] rounded text-red-500 hover:text-red-400 hover:border-red-500 transition-colors">
              <Plus size={14} /> Add Sell Rule
            </button>
          </div>
        )}
      </div>

      {/* EXECUTION SETTINGS SECTION */}
      <div className="bg-[#1e222d] border border-[#2b2b2b] rounded-lg overflow-hidden">
        <div 
          className="p-3 bg-[#131722] border-b border-[#2b2b2b] flex justify-between items-center cursor-pointer select-none"
          onClick={() => setExpanded(prev => ({...prev, execution: !prev.execution}))}
        >
          <span className="font-semibold text-purple-500">4. Execution Settings</span>
          {expanded.execution ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
        {expanded.execution && (
          <div className="p-3 space-y-4">
            <label className="flex items-center gap-2 text-gray-300 cursor-pointer">
              <input 
                type="checkbox" 
                checked={strategy.options?.enabled || false}
                onChange={e => updateOptions({ enabled: e.target.checked })}
                className="w-4 h-4 rounded bg-[#1e222d] border-[#2b2b2b] focus:ring-purple-500"
              />
              Trade Options (Black-Scholes Model)
            </label>
            
            {strategy.options?.enabled && (
              <div className="grid grid-cols-2 gap-3 p-3 bg-[#131722] rounded border border-[#2b2b2b]">
                <div className="flex flex-col gap-1 col-span-2 sm:col-span-1">
                  <label className="text-gray-500">Days to Expiry (DTE)</label>
                  <div className="bg-[#1e222d] border border-[#2b2b2b] text-purple-400/80 rounded px-2 py-1 text-[10px] sm:text-xs flex items-center h-[26px]">
                    Auto-calculated to next Thu/Tue expiry
                  </div>
                </div>
                <div className="flex flex-col gap-1 col-span-2 sm:col-span-1">
                  <label className="text-gray-500">Implied Vol. (e.g. 0.15)</label>
                  <input 
                    type="number" 
                    step="0.01"
                    value={strategy.options?.iv ?? 0.15}
                    onChange={e => updateOptions({ iv: Number(e.target.value) })}
                    className="bg-[#1e222d] border border-[#2b2b2b] text-white rounded px-2 h-[26px] outline-none focus:border-purple-500 font-mono"
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <datalist id="indicatorsList">
        {indicatorOptions.map((opt: string) => <option key={opt} value={opt} />)}
      </datalist>
    </div>
  );
}
