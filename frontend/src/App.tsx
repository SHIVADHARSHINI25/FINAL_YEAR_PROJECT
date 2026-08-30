import { useState } from 'react';
import { Activity, LayoutDashboard, Database, History, PlayCircle } from 'lucide-react';
import SetupPage from './pages/SetupPage';
import MonitorPage from './pages/MonitorPage';
import HistoryPage from './pages/HistoryPage';
import KnowledgePage from './pages/KnowledgePage';

function App() {
  const [activeTab, setActiveTab] = useState('setup');
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);

  const renderContent = () => {
    switch (activeTab) {
      case 'setup':
        return <SetupPage onStart={(runId) => { setCurrentRunId(runId); setActiveTab('monitor'); }} />;
      case 'monitor':
        return <MonitorPage runId={currentRunId} />;
      case 'history':
        return <HistoryPage onViewRun={(runId) => { setCurrentRunId(runId); setActiveTab('monitor'); }} />;
      case 'knowledge':
        return <KnowledgePage />;
      default:
        return <SetupPage onStart={(runId) => { setCurrentRunId(runId); setActiveTab('monitor'); }} />;
    }
  };

  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Sidebar */}
      <aside className="w-64 bg-[#1e293b]/80 backdrop-blur-md border-r border-slate-700/50 flex flex-col transition-all duration-300">
        <div className="p-6">
          <div className="flex items-center gap-3 text-indigo-400 mb-2">
            <Activity className="w-8 h-8" />
            <h1 className="text-xl font-bold tracking-tight text-white">NeuroHeal</h1>
          </div>
          <p className="text-xs text-slate-400 font-medium tracking-wide uppercase">Autonomous ML Pipelines</p>
        </div>

        <nav className="flex-1 px-4 space-y-2 mt-4">
          <NavItem 
            icon={<PlayCircle className="w-5 h-5" />} 
            label="New Pipeline" 
            isActive={activeTab === 'setup'} 
            onClick={() => setActiveTab('setup')} 
          />
          <NavItem 
            icon={<LayoutDashboard className="w-5 h-5" />} 
            label="Live Monitor" 
            isActive={activeTab === 'monitor'} 
            onClick={() => setActiveTab('monitor')} 
          />
          <NavItem 
            icon={<History className="w-5 h-5" />} 
            label="Run History" 
            isActive={activeTab === 'history'} 
            onClick={() => setActiveTab('history')} 
          />
          <NavItem 
            icon={<Database className="w-5 h-5" />} 
            label="Knowledge Base" 
            isActive={activeTab === 'knowledge'} 
            onClick={() => setActiveTab('knowledge')} 
          />
        </nav>
        
        <div className="p-4 border-t border-slate-700/50 text-xs text-slate-500 text-center">
          v1.0.0-prototype
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto relative">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-900/20 via-slate-900/0 to-slate-900/0 pointer-events-none"></div>
        <div className="relative z-10 p-8 min-h-full">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

function NavItem({ icon, label, isActive, onClick }: { icon: React.ReactNode, label: string, isActive: boolean, onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
        isActive 
          ? 'bg-indigo-500/10 text-indigo-300 border border-indigo-500/20' 
          : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border border-transparent'
      }`}
    >
      <div className={`${isActive ? 'text-indigo-400' : 'text-slate-500 group-hover:text-slate-300'} transition-colors duration-200`}>
        {icon}
      </div>
      <span className="font-medium">{label}</span>
    </button>
  );
}

export default App;
