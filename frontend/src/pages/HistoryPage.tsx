import { useEffect, useState } from 'react';
import { History, Eye, Activity } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const API_BASE = window.location.hostname === 'localhost' && window.location.port === '5173' ? 'http://localhost:8000' : '';

export default function HistoryPage({ onViewRun }: { onViewRun: (runId: string) => void }) {
  const [runs, setRuns] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/pipelines`)
      .then(res => res.json())
      .then(data => {
        setRuns(data);
        setIsLoading(false);
      })
      .catch(err => {
        console.error(err);
        setIsLoading(false);
      });
  }, []);

  return (
    <div className="max-w-6xl mx-auto animation-fade-in mt-6">
      <h2 className="text-3xl font-semibold mb-8 flex items-center gap-3">
        <History className="w-8 h-8 text-indigo-400" />
        Pipeline Run History
      </h2>

      {!isLoading && runs.length > 0 && (
        <div className="bg-[#1e293b]/60 border border-slate-700/50 p-6 rounded-2xl shadow-xl mb-8">
          <h3 className="font-semibold text-lg mb-4 text-slate-200 flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-400" />
            Pipeline Status Trends
          </h3>
          <div style={{ width: '100%', height: 250 }}>
            <ResponsiveContainer>
              <BarChart
                data={Object.values(runs.reduce((acc: any, curr: any) => {
                  const date = new Date(curr.start_time + 'Z').toLocaleDateString();
                  if (!acc[date]) acc[date] = { date, success: 0, recovered: 0, failed: 0, error: 0 };
                  if (curr.status === 'success') acc[date].success++;
                  else if (curr.status === 'recovered') acc[date].recovered++;
                  else if (curr.status === 'failed') acc[date].failed++;
                  else if (curr.status === 'error') acc[date].error++;
                  return acc;
                }, {}))}
                margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} allowDecimals={false} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} cursor={{fill: '#334155', opacity: 0.4}} />
                <Legend />
                <Bar dataKey="success" stackId="a" fill="#10b981" />
                <Bar dataKey="recovered" stackId="a" fill="#f59e0b" />
                <Bar dataKey="failed" stackId="a" fill="#f43f5e" />
                <Bar dataKey="error" stackId="a" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="flex justify-center p-12">
          <div className="w-8 h-8 border-2 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="bg-[#1e293b]/60 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-xl overflow-x-auto">
          <table className="w-full text-sm text-left whitespace-nowrap">
            <thead className="text-xs uppercase bg-slate-800/80 text-slate-400">
              <tr>
                <th className="px-6 py-4">Run ID</th>
                <th className="px-6 py-4">Dataset</th>
                <th className="px-6 py-4">Status</th>
                <th className="px-6 py-4">Started At</th>
                <th className="px-6 py-4">Recovery Strategy</th>
                <th className="px-6 py-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run: any) => (
                <tr key={run.run_id} className="border-b border-slate-700/50 hover:bg-slate-800/30 transition-colors">
                  <td className="px-6 py-4 font-mono text-xs text-indigo-300">{run.run_id.substring(0, 8)}...</td>
                  <td className="px-6 py-4 text-slate-300">{run.dataset_name}</td>
                  <td className="px-6 py-4">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium border ${
                      run.status === 'success' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                      run.status === 'recovered' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                      run.status === 'failed' || run.status === 'error' ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' :
                      'bg-slate-500/10 text-slate-400 border-slate-500/20'
                    }`}>
                      {run.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-slate-400">
                    {new Date(run.start_time + 'Z').toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-slate-300">
                    {run.recovery_used || <span className="text-slate-600">—</span>}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <button 
                      onClick={() => onViewRun(run.run_id)}
                      className="inline-flex items-center gap-1.5 text-indigo-400 hover:text-indigo-300 bg-indigo-500/10 hover:bg-indigo-500/20 px-3 py-1.5 rounded-lg transition-colors"
                    >
                      <Eye className="w-4 h-4" /> View
                    </button>
                  </td>
                </tr>
              ))}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-slate-500">
                    No pipeline runs found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
