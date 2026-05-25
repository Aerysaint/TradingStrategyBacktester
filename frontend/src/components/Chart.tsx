'use client';

import { useEffect, useRef, useImperativeHandle, forwardRef, useMemo } from 'react';
import { createChart, ColorType, CrosshairMode, IChartApi, ISeriesApi, SeriesType, CandlestickSeries, LineSeries, HistogramSeries, createSeriesMarkers } from 'lightweight-charts';

export interface ChartProps {
  data: any[];
  markers?: any[];
  indicators?: any[];
  indicatorDefs?: any[];
  colors?: {
    backgroundColor?: string;
    lineColor?: string;
    textColor?: string;
  };
}

export interface ChartRef {
  getMainChart: () => IChartApi | null;
}

const OSCILLATORS = ['rsi', 'macd', 'stoch'];

export const LightweightChart = forwardRef<ChartRef, ChartProps>(
  (
    {
      data,
      markers = [],
      indicators = [],
      indicatorDefs = [],
      colors: {
        backgroundColor = '#1e1e1e',
        textColor = '#d1d4dc',
      } = {},
    },
    ref
  ) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartsMap = useRef<{ [id: string]: IChartApi }>({});
    const mainSeriesRef = useRef<ISeriesApi<any> | null>(null);
    const indSeriesMap = useRef<{ [id: string]: ISeriesApi<any> }>({});
    const markersPluginRef = useRef<any>(null);
    
    // Memoize the list of oscillators so we know how many panes to create
    const oscillators = useMemo(() => {
      return (indicatorDefs || []).filter(d => OSCILLATORS.includes(d.name.toLowerCase()));
    }, [indicatorDefs]);

    // 1. Initialise all charts
    useEffect(() => {
      if (!containerRef.current) return;
      
      const elements: { [id: string]: HTMLElement } = {
         main: containerRef.current.querySelector('#chart-main') as HTMLElement,
      };
      oscillators.forEach(osc => {
         const el = containerRef.current!.querySelector(`#chart-${osc.id}`) as HTMLElement;
         if (el) elements[osc.id] = el;
      });

      const charts: { [id: string]: IChartApi } = {};
      
      const commonOptions = {
        layout: { background: { type: ColorType.Solid, color: backgroundColor }, textColor },
        grid: { vertLines: { color: '#2b2b2b' }, horzLines: { color: '#2b2b2b' } },
        crosshair: { mode: CrosshairMode.Normal },
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
        },
      };

      // Create main chart
      if (elements.main) {
        charts['main'] = createChart(elements.main, {
          ...commonOptions,
          width: elements.main.clientWidth,
          height: elements.main.clientHeight,
        });
        
        mainSeriesRef.current = charts['main'].addSeries(CandlestickSeries, {
          upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350',
        });
        
        markersPluginRef.current = createSeriesMarkers(mainSeriesRef.current, []);
      }

      // Create oscillator charts
      oscillators.forEach(osc => {
         if (elements[osc.id]) {
            charts[osc.id] = createChart(elements[osc.id], {
              ...commonOptions,
              width: elements[osc.id].clientWidth,
              height: elements[osc.id].clientHeight,
            });
         }
      });
      
      chartsMap.current = charts;

      // Sync time scales across panes smoothly
      const chartIds = Object.keys(charts);
      let isSyncing = false;
      
      chartIds.forEach(sourceId => {
         const sourceChart = charts[sourceId];
         sourceChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
             if (isSyncing || !range) return;
             isSyncing = true;
             chartIds.forEach(targetId => {
                 if (targetId !== sourceId) {
                     charts[targetId].timeScale().setVisibleLogicalRange(range);
                 }
             });
             isSyncing = false;
         });
      });

      const handleResize = () => {
         Object.keys(charts).forEach(id => {
             const el = id === 'main' ? elements.main : elements[id];
             if (el) {
                 charts[id].applyOptions({ width: el.clientWidth, height: el.clientHeight });
             }
         });
      };
      window.addEventListener('resize', handleResize);

      return () => {
         window.removeEventListener('resize', handleResize);
         Object.values(charts).forEach(c => c.remove());
         chartsMap.current = {};
         indSeriesMap.current = {};
         mainSeriesRef.current = null;
      };
    }, [backgroundColor, textColor, oscillators]);

    // 2. Set Chart data
    useEffect(() => {
      if (mainSeriesRef.current && data.length > 0) {
        // Critical Fix: Filter out data points with missing/NaN timestamps which disrupt format mapping 
        const validData = data.filter(d => 
            !isNaN(d.time) && 
            d.open != null && !isNaN(d.open) &&
            d.high != null && !isNaN(d.high) &&
            d.low != null && !isNaN(d.low) &&
            d.close != null && !isNaN(d.close)
        );
        const sortedData = [...validData].sort((a, b) => a.time - b.time);
        
        // Final sanity check: Strictly unique timestamps required by Lightweight Charts
        const uniqueData = sortedData.filter((item, i, arr) => 
            i === 0 || item.time !== arr[i - 1].time
        );
        try {
            mainSeriesRef.current.setData(uniqueData);
        } catch (e) {
            console.error("Failed to set main series data:", e);
        }
      }
    }, [data, oscillators]); // Note: oscillators dependency causes chart re-population after pane recreation 

    // 3. Set Markers
    useEffect(() => {
      if (markersPluginRef.current) {
        const sortedMarkers = [...markers].sort((a, b) => a.time - b.time);
        try {
          markersPluginRef.current.setMarkers(sortedMarkers);
        } catch (e) {
          console.warn("Could not set markers:", e);
        }
      }
    }, [markers, oscillators]);

    // 4. Draw Indicators
    useEffect(() => {
      if (!chartsMap.current['main']) return;

      const colorsPattern = ['#2962FF', '#FF6D00', '#00C853', '#00BFA5', '#AA00FF', '#C51162'];
      
      // Clean stale indicator lines
      const currentIndicatorIds = new Set(indicators.map(ind => ind.id));
      Object.keys(indSeriesMap.current).forEach(id => {
          if (!currentIndicatorIds.has(id)) {
              const oscBaseId = oscillators.find(o => id.startsWith(o.id))?.id || 'main';
              const targetChart = chartsMap.current[oscBaseId] || chartsMap.current['main'];
              if (targetChart && indSeriesMap.current[id]) {
                 targetChart.removeSeries(indSeriesMap.current[id]);
              }
              delete indSeriesMap.current[id];
          }
      });
      
      indicators.forEach((ind, i) => {
          // Which pane does this go to?
          let targetPaneId = 'main';
          const matchOsc = oscillators.find(o => ind.id.startsWith(o.id));
          if (matchOsc) targetPaneId = matchOsc.id;
          
          const targetChart = chartsMap.current[targetPaneId];
          if (!targetChart) return;
          
          let series = indSeriesMap.current[ind.id];
          if (!series) {
              const isHistogram = ind.id.toLowerCase().includes('macdh');
              // Setup correct UI styles for special overlays 
              if (isHistogram) {
                  series = targetChart.addSeries(HistogramSeries, {
                      color: '#26a69a',
                      title: 'MACD Histogram'
                  });
              } else {
                  series = targetChart.addSeries(LineSeries, {
                      color: colorsPattern[i % colorsPattern.length],
                      lineWidth: 2,
                      title: ind.id
                  });
              }
              indSeriesMap.current[ind.id] = series;
          }
          
          // Format Data values ensuring strict sorting and unique timestamps
          const validIndData = ind.data.filter((d: any) => !isNaN(d.value) && d.value !== null);
          const formattedIndData = validIndData.map((d: any) => ({
             time: (new Date(d.time + "Z")).getTime() / 1000,
             value: d.value
          }))
          .filter((d:any) => !isNaN(d.time))
          .sort((a: any, b: any) => a.time - b.time)
          .filter((item: any, idx: number, arr: any[]) => idx === 0 || item.time !== arr[idx - 1].time);
          
          // Format histogram colors based on positive/negative
          if (ind.id.toLowerCase().includes('macdh')) {
              formattedIndData.forEach((d: any) => {
                  d.color = d.value >= 0 ? 'rgba(38, 166, 154, 0.8)' : 'rgba(239, 83, 80, 0.8)';
              });
          }
          
          try {
             if (formattedIndData.length > 0) {
                 series.setData(formattedIndData);
             }
          } catch(e) {
             console.warn("Could not set indicator data for", ind.id, e);
          }
      });
      
    }, [indicators, oscillators]);

    useImperativeHandle(ref, () => ({
      getMainChart: () => chartsMap.current['main'] || null,
    }));

    return (
        <div ref={containerRef} className="w-full h-full flex flex-col overflow-hidden">
            <div id="chart-main" className="flex-1 w-full" />
            {oscillators.map(osc => (
                <div key={osc.id} id={`chart-${osc.id}`} className="w-full h-[200px] border-t border-[#2b2b2b]" />
            ))}
        </div>
    );
  }
);

LightweightChart.displayName = 'LightweightChart';
