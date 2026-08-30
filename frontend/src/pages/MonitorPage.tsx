import { useEffect, useState } from 'react';
import { CheckCircle2, XCircle, Loader2, AlertTriangle } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import AgentNetwork from '../components/AgentNetwork';

const API_BASE = window.location.hostname === 'localhost' && window.location.port === '5173' ? 'http://localhost:8000' : '';
const WS_BASE = window.location.hostname === 'localhost' && window.location.port === '5173' ? 'ws://localhost:8000' : (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host;

export default function MonitorPage({ runId }: { runId: string | null }) {
  const [events, setEvents] = useState<any[]>([]);
  const [status, setStatus] = useState<string>('waiting');
  
  useEffect(() => {
    if (!runId) return;
    
    // Fetch initial state and historical failures (in case we are viewing a past run)
    fetch(`${API_BASE}/pipelines/${runId}`)
      .then(res => res.json())
      .then(data => {
        if (data.status) setStatus(data.status);
      })
      .catch(err => console.error(err));

    fetch(`${API_BASE}/pipelines/${runId}/failures`)
      .then(res => res.json())
      .then(data => {
        if (data.detection && data.detection.length > 0) {
          const historicalEvents: any[] = [
            { type: 'failure_detected', data: { evidence: data.detection } },
            { type: 'decision_engine_ranked', data: { ranked: data.ranked_strategies } },
            { type: 'verification_completed', data: { results: data.verification_results, all_passed: data.recovery_action?.success } }
          ];
          if (data.final_metrics) {
            historicalEvents.push({ type: 'run_completed', data: { status: 'recovered', metrics: data.final_metrics, samples: data.final_samples, target_col: data.final_target_col || 'label' } });
          }
          setEvents(prev => {
            if (prev.some(e => e.type === 'failure_detected')) return prev;
            return [...prev, ...historicalEvents];
          });
        } else if (data.final_metrics) {
          // If no failure, but has metrics (success path)
          setEvents(prev => {
            if (prev.some(e => e.type === 'run_completed')) return prev;
            return [...prev, { type: 'run_completed', data: { status: 'success', metrics: data.final_metrics, samples: data.final_samples, target_col: data.final_target_col || 'label' } }];
          });
        }
      })
      .catch(err => console.error(err));
    
    const ws = new WebSocket(`${WS_BASE}/pipelines/${runId}/live`);
    
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'current_status' || msg.type === 'run_completed') {
        setStatus(msg.data.status);
      }
      setEvents((prev) => [...prev, msg]);
    };
    
    return () => ws.close();
  }, [runId]);

  if (!runId) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-slate-500">
        <AlertTriangle className="w-12 h-12 mb-4 opacity-50" />
        <p>No active pipeline run. Start one from the New Pipeline tab.</p>
      </div>
    );
  }

  // Look for failure details in events
  const failureDetected = events.find(e => e.type === 'failure_detected');
  const decisionRanked = events.find(e => e.type === 'decision_engine_ranked');
  const verification = events.find(e => e.type === 'verification_completed');
  const runCompletedEvent = events.find(e => e.type === 'run_completed');

  return (
    <div className="space-y-6 animation-fade-in max-w-6xl mx-auto">
      <div className="flex items-center justify-between bg-[#1e293b]/80 p-6 rounded-2xl border border-slate-700/50">
        <div>
          <h2 className="text-2xl font-bold mb-1 flex items-center gap-2">
            Pipeline Monitor
            {status === 'running' && <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />}
            {status === 'success' && <CheckCircle2 className="w-6 h-6 text-emerald-400" />}
            {status === 'recovered' && <CheckCircle2 className="w-6 h-6 text-amber-400" />}
            {status === 'failed' && <XCircle className="w-6 h-6 text-rose-400" />}
          </h2>
          <p className="text-sm text-slate-400">Run ID: <span className="font-mono text-slate-300">{runId}</span></p>
        </div>
        <div className="flex items-center gap-4">
          {(status === 'success' || status === 'recovered') && (
            <>
              <a 
                href={`${API_BASE}/pipelines/${runId}/dataset`}
                download
                className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm transition-colors shadow-lg shadow-emerald-500/20"
              >
                Download Fixed Dataset (.csv)
              </a>
              <a 
                href={`${API_BASE}/pipelines/${runId}/model`}
                download
                className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm transition-colors shadow-lg shadow-indigo-500/20"
              >
                Download Model (.pkl)
              </a>
            </>
          )}
          <div className="px-4 py-2 rounded-lg bg-slate-900 border border-slate-700 font-medium uppercase tracking-wider text-sm">
            {status}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Event Log */}
        <div className="lg:col-span-1 bg-[#1e293b]/60 rounded-2xl border border-slate-700/50 p-6 overflow-hidden flex flex-col h-[600px]">
          <h3 className="font-semibold text-lg mb-4 text-slate-200">Execution Log</h3>
          <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin">
            {events.filter(e => e.type !== 'resource_metrics' && e.type !== 'training_metrics').map((ev, i) => (
              <div key={i} className="text-sm p-3 rounded-lg bg-slate-900/50 border border-slate-700/30">
                <div className="text-indigo-400 font-mono text-xs mb-1">{ev.type}</div>
                <div className="text-slate-300 break-words font-mono text-[11px] opacity-80">
                  {JSON.stringify(ev.data).substring(0, 100)}...
                </div>
              </div>
            ))}
            {events.length === 0 && <div className="text-slate-500 text-sm">Waiting for events...</div>}
          </div>
        </div>

        {/* Detailed Failure & Recovery View */}
        <div className="lg:col-span-2 space-y-6">
          {/* Agent Network Visualisation */}
          <div className="bg-[#1e293b]/60 rounded-2xl border border-slate-700/50 p-6">
            <h3 className="font-semibold text-lg mb-4 text-slate-200">Agent Interaction Network</h3>
            <AgentNetwork activeEvent={events.length > 0 ? events[events.length - 1].type : ''} />
          </div>

          {failureDetected && (
            <div className="bg-rose-950/20 border border-rose-900/50 rounded-2xl p-6 animation-fade-in">
              <h3 className="text-xl font-semibold text-rose-400 mb-4 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6" /> Failure Detected
              </h3>
              <div className="space-y-4">
                {failureDetected.data.evidence.map((ev: any, i: number) => (
                  <div key={i} className="bg-rose-900/10 p-4 rounded-xl border border-rose-800/30">
                    <div className="font-medium text-rose-300 mb-2">Agent: {ev.agent_name}</div>
                    <p className="text-slate-300 text-sm">{ev.finding}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {decisionRanked && (
            <div className="bg-[#1e293b]/80 border border-slate-700/50 rounded-2xl p-6 animation-fade-in">
              <h3 className="text-xl font-semibold text-indigo-400 mb-4">Decision Engine Ranking</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs uppercase bg-slate-800/50 text-slate-400">
                    <tr>
                      <th className="px-4 py-3 rounded-tl-lg">Strategy</th>
                      <th className="px-4 py-3">Total Score</th>
                      <th className="px-4 py-3">Confidence</th>
                      <th className="px-4 py-3">History</th>
                      <th className="px-4 py-3">Accuracy</th>
                      <th className="px-4 py-3 rounded-tr-lg">Risk</th>
                    </tr>
                  </thead>
                  <tbody>
                    {decisionRanked.data.ranked.map((strategy: any, i: number) => (
                      <tr key={i} className={`border-b border-slate-700/50 ${i === 0 ? 'bg-indigo-900/20' : ''}`}>
                        <td className="px-4 py-4 font-medium text-slate-200">
                          {strategy.strategy_name}
                          {i === 0 && <span className="ml-2 text-[10px] bg-indigo-500 text-white px-2 py-0.5 rounded-full uppercase tracking-wider">Selected</span>}
                        </td>
                        <td className="px-4 py-4 font-mono text-indigo-300">{strategy.total_score.toFixed(2)}</td>
                        <td className="px-4 py-4">{strategy.confidence_score.toFixed(2)}</td>
                        <td className="px-4 py-4">{strategy.historical_success_rate.toFixed(2)}</td>
                        <td className="px-4 py-4">{strategy.expected_accuracy_preservation.toFixed(2)}</td>
                        <td className="px-4 py-4">{strategy.risk_level.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {verification && (
            <div className={`p-6 rounded-2xl border ${verification.data.all_passed ? 'bg-emerald-950/20 border-emerald-900/50' : 'bg-rose-950/20 border-rose-900/50'} animation-fade-in`}>
               <h3 className={`text-xl font-semibold mb-4 flex items-center gap-2 ${verification.data.all_passed ? 'text-emerald-400' : 'text-rose-400'}`}>
                {verification.data.all_passed ? <CheckCircle2 className="w-6 h-6" /> : <XCircle className="w-6 h-6" />}
                Verification Phase
              </h3>
              <div className="space-y-3">
                {verification.data.results.map((res: any, i: number) => (
                  <div key={i} className="flex items-start gap-3 bg-slate-900/30 p-3 rounded-lg border border-slate-700/30">
                    {res.passed ? <CheckCircle2 className="w-5 h-5 text-emerald-500 shrink-0" /> : <XCircle className="w-5 h-5 text-rose-500 shrink-0" />}
                    <div>
                      <div className="font-medium text-slate-200">{res.check_name}</div>
                      <div className="text-sm text-slate-400">{res.details}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* FINAL OUTPUT PANEL */}
      {runCompletedEvent && runCompletedEvent.data.metrics && (
        <div className="bg-[#1e293b]/60 border border-slate-700/50 p-6 rounded-2xl shadow-xl mt-6">
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <CheckCircle2 className="w-6 h-6 text-emerald-400" />
              Model Evaluation & Performance
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
              {/* Metrics Bar Chart */}
              <div>
                <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">Performance Metrics</h4>
                <div style={{ width: '100%', height: 250 }}>
                  <ResponsiveContainer>
                    <BarChart
                      data={Object.entries(runCompletedEvent.data.metrics).map(([name, value]) => ({
                        name: name.replace('_', ' '),
                        value: Number(value)
                      }))}
                      margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                      layout="vertical"
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={true} vertical={false} />
                      <XAxis type="number" stroke="#94a3b8" fontSize={11} />
                      <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={11} width={80} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} cursor={{fill: '#334155', opacity: 0.4}} />
                      <Bar dataKey="value" fill="#10b981" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Actual vs Predicted Line Chart */}
              <div>
                <h4 className="text-sm font-medium text-slate-400 uppercase tracking-wider mb-4">Actual vs Predicted</h4>
                <div style={{ width: '100%', height: 250 }}>
                  <ResponsiveContainer>
                    <LineChart
                      data={runCompletedEvent.data.samples?.map((s: any, i: number) => ({
                        sample: `S${i+1}`,
                        Actual: s.actual,
                        Predicted: s.prediction
                      })) || []}
                      margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                      <XAxis dataKey="sample" stroke="#94a3b8" fontSize={11} />
                      <YAxis stroke="#94a3b8" fontSize={11} />
                      <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                      <Legend />
                      <Line type="monotone" dataKey="Actual" stroke="#818cf8" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                      <Line type="monotone" dataKey="Predicted" stroke="#f43f5e" strokeWidth={2} dot={{ r: 4 }} strokeDasharray="5 5" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-700/50">
              <div>
                <h4 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">Raw Metrics</h4>
                <div className="grid grid-cols-2 gap-3">
                  {Object.entries(runCompletedEvent.data.metrics).map(([k, v]) => (
                    <div key={k} className="bg-slate-900/50 px-3 py-2 rounded-xl border border-slate-700/50 flex justify-between items-center">
                      <span className="text-xs text-slate-400 capitalize">{k.replace('_', ' ')}</span>
                      <span className="text-sm font-bold text-emerald-400">{Number(v).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-3">Sample Outputs</h4>
                <div className="space-y-2">
                  {runCompletedEvent.data.samples?.slice(0, 3).map((s: any, idx: number) => {
                    const inputStr = Object.entries(s.input).map(([k,v]) => `${k}=${v}`).join(', ');
                    return (
                      <div key={idx} className="bg-slate-900/50 p-2.5 rounded-xl border border-slate-700/50 text-xs flex justify-between items-center">
                        <div className="text-slate-400 truncate max-w-[150px] xl:max-w-[200px]" title={inputStr}>
                          {inputStr}
                        </div>
                        <div className="flex gap-3 shrink-0">
                          <span className="text-slate-500">Act: <span className="text-slate-300">{s.actual}</span></span>
                          <span className="text-slate-500">Pred: <span className="text-indigo-400">{s.prediction}</span></span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
    </div>
  );
}
