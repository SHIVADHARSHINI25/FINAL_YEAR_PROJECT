import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

export default function LiveTelemetry({ events }: { events: any[] }) {
  // Extract resource metrics
  const resourceMetrics = events
    .filter(e => e.type === 'resource_metrics')
    .map(e => e.data);

  // Extract training metrics
  const trainingMetrics = events
    .filter(e => e.type === 'training_metrics')
    .map(e => e.data);

  // Extract decision engine ranking
  const deRanked = events.find(e => e.type === 'decision_engine_ranked');
  
  let radarData: any[] = [];
  if (deRanked && deRanked.data && deRanked.data.ranked) {
    // Take top 3 strategies for radar chart
    radarData = deRanked.data.ranked.slice(0, 3).map((r: any) => ({
      subject: r.strategy_name.replace(/_/g, ' '),
      Confidence: Math.round(r.confidence * 100),
      History: Math.round(r.historical_success * 100),
      Risk: Math.round(r.risk_score * 100),
      Total: Math.round(r.total_score * 100),
    }));
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
      
      {/* Resource Monitor */}
      <div className="bg-[#1e293b]/60 rounded-2xl border border-slate-700/50 p-6">
        <h3 className="font-semibold text-lg mb-4 text-slate-200">Resource Utilization (CPU/GPU)</h3>
        <div style={{ width: '100%', height: 250 }}>
          <ResponsiveContainer>
            <LineChart data={resourceMetrics} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="time" stroke="#94a3b8" fontSize={10} />
              <YAxis domain={[0, 100]} stroke="#94a3b8" fontSize={10} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
              <Line type="monotone" dataKey="cpu_percent" name="CPU %" stroke="#818cf8" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="gpu_memory_percent" name="GPU %" stroke="#f43f5e" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Training Loss */}
      <div className="bg-[#1e293b]/60 rounded-2xl border border-slate-700/50 p-6">
        <h3 className="font-semibold text-lg mb-4 text-slate-200">Training Loss Curve</h3>
        <div style={{ width: '100%', height: 250 }}>
          <ResponsiveContainer>
            <AreaChart data={trainingMetrics} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={10} />
              <YAxis stroke="#94a3b8" fontSize={10} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
              <Area type="monotone" dataKey="loss" stroke="#10b981" fillOpacity={1} fill="url(#colorLoss)" isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Decision Engine Radar */}
      <div className="bg-[#1e293b]/60 rounded-2xl border border-slate-700/50 p-6">
        <h3 className="font-semibold text-lg mb-4 text-slate-200">Decision Engine Scores</h3>
        {radarData.length > 0 ? (
          <div style={{ width: '100%', height: 250 }}>
            <ResponsiveContainer>
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                <Radar name="Total Score" dataKey="Total" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="flex items-center justify-center h-[250px] text-slate-500 italic text-sm">
            Waiting for Decision Engine evaluation...
          </div>
        )}
      </div>

    </div>
  );
}
