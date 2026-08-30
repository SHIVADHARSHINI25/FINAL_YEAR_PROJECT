import { useEffect, useState } from 'react';
import { Database, Lightbulb, CheckCircle2, AlertTriangle, HelpCircle } from 'lucide-react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const API_BASE = window.location.hostname === 'localhost' && window.location.port === '5173' ? 'http://localhost:8000' : '';

export default function KnowledgePage() {
  const [entries, setEntries] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchKnowledge = async () => {
      try {
        const res = await fetch(`${API_BASE}/knowledge`);
        const data = await res.json();
        setEntries(data);
        setIsLoading(false);
      } catch (err) {
        console.error(err);
        setIsLoading(false);
      }
    };
    fetchKnowledge();
  }, []);

  // Compute aggregated stats
  const total = entries.length;
  const successes = entries.filter(e => e.outcome === 'success').length;
  const partials = entries.filter(e => e.outcome === 'partial').length;
  const successRate = total > 0 ? (successes + 0.5 * partials) / total : 0;

  return (
    <div className="max-w-6xl mx-auto animation-fade-in mt-6">
      <h2 className="text-3xl font-semibold mb-8 flex items-center gap-3">
        <Database className="w-8 h-8 text-indigo-400" />
        Knowledge Repository
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-[#1e293b]/60 border border-slate-700/50 p-6 rounded-2xl flex items-center gap-4">
          <div className="bg-indigo-500/20 p-4 rounded-xl text-indigo-400">
            <Lightbulb className="w-8 h-8" />
          </div>
          <div>
            <div className="text-slate-400 text-sm font-medium">Total Entries</div>
            <div className="text-3xl font-bold text-slate-200">{total}</div>
          </div>
        </div>
        <div className="bg-[#1e293b]/60 border border-slate-700/50 p-6 rounded-2xl flex items-center gap-4">
          <div className="bg-emerald-500/20 p-4 rounded-xl text-emerald-400">
            <CheckCircle2 className="w-8 h-8" />
          </div>
          <div>
            <div className="text-slate-400 text-sm font-medium">Historical Success Rate</div>
            <div className="text-3xl font-bold text-slate-200">{(successRate * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
      
      {/* Knowledge Base Graphs */}
      {entries.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-[#1e293b]/60 border border-slate-700/50 p-6 rounded-2xl shadow-xl">
            <h3 className="font-semibold text-lg mb-4 text-slate-200">Root Cause Distribution</h3>
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <PieChart>
                  <Pie
                    data={Object.entries(entries.reduce((acc, curr) => {
                      acc[curr.root_cause_type] = (acc[curr.root_cause_type] || 0) + 1;
                      return acc;
                    }, {} as any)).map(([name, value]) => ({ name, value }))}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {Object.keys(entries.reduce((acc, curr) => {
                      acc[curr.root_cause_type] = (acc[curr.root_cause_type] || 0) + 1;
                      return acc;
                    }, {})).map((_, index) => {
                      const colors = ['#818cf8', '#f43f5e', '#10b981', '#f59e0b', '#8b5cf6'];
                      return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />;
                    })}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="bg-[#1e293b]/60 border border-slate-700/50 p-6 rounded-2xl shadow-xl">
            <h3 className="font-semibold text-lg mb-4 text-slate-200">Strategy Success Rates</h3>
            <div style={{ width: '100%', height: 300 }}>
              <ResponsiveContainer>
                <BarChart
                  data={Object.values(entries.reduce((acc, curr) => {
                    if (!acc[curr.recovery_strategy_used]) {
                      acc[curr.recovery_strategy_used] = { name: curr.recovery_strategy_used.replace(/_/g, ' '), success: 0, partial: 0, failure: 0 };
                    }
                    acc[curr.recovery_strategy_used][curr.outcome]++;
                    return acc;
                  }, {} as any))}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} tickFormatter={(val) => val.length > 15 ? val.substring(0, 15) + '...' : val} />
                  <YAxis stroke="#94a3b8" fontSize={12} allowDecimals={false} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} cursor={{fill: '#334155', opacity: 0.4}} />
                  <Legend />
                  <Bar dataKey="success" stackId="a" fill="#10b981" />
                  <Bar dataKey="partial" stackId="a" fill="#f59e0b" />
                  <Bar dataKey="failure" stackId="a" fill="#f43f5e" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="flex justify-center p-12">
          <div className="w-8 h-8 border-2 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="bg-[#1e293b]/60 backdrop-blur-xl border border-slate-700/50 rounded-2xl shadow-xl overflow-hidden">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase bg-slate-800/80 text-slate-400">
              <tr>
                <th className="px-6 py-4">Failure Signature</th>
                <th className="px-6 py-4">Root Cause Type</th>
                <th className="px-6 py-4">Recovery Strategy</th>
                <th className="px-6 py-4">Outcome</th>
                <th className="px-6 py-4">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry: any, i: number) => (
                <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-800/30 transition-colors">
                  <td className="px-6 py-4 font-mono text-xs text-rose-300">{entry.failure_signature}</td>
                  <td className="px-6 py-4 text-slate-300">{entry.root_cause_type}</td>
                  <td className="px-6 py-4 text-indigo-300 font-medium">{entry.recovery_strategy_used}</td>
                  <td className="px-6 py-4">
                    <span className={`flex items-center gap-1.5 w-fit px-2.5 py-1 rounded-full text-xs font-medium border ${
                      entry.outcome === 'success' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
                      entry.outcome === 'partial' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' :
                      'bg-rose-500/10 text-rose-400 border-rose-500/20'
                    }`}>
                      {entry.outcome === 'success' && <CheckCircle2 className="w-3.5 h-3.5" />}
                      {entry.outcome === 'partial' && <AlertTriangle className="w-3.5 h-3.5" />}
                      {entry.outcome === 'failure' && <HelpCircle className="w-3.5 h-3.5" />}
                      {entry.outcome}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-slate-400">
                    {new Date(entry.timestamp + 'Z').toLocaleString()}
                  </td>
                </tr>
              ))}
              {entries.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-6 py-8 text-center text-slate-500">
                    No knowledge base entries found.
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
