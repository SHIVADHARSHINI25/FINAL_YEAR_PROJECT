import { useState } from 'react';
import { UploadCloud, Play, Settings2 } from 'lucide-react';

const API_BASE = window.location.hostname === 'localhost' && window.location.port === '5173' ? 'http://localhost:8000' : '';

export default function SetupPage({ onStart }: { onStart: (runId: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [algorithm, setAlgorithm] = useState('random_forest');
  const [targetCol, setTargetCol] = useState('label');
  const [isLoading, setIsLoading] = useState(false);
  const [injectFailure, setInjectFailure] = useState(true);

  const handleStart = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('algorithm', algorithm);
    formData.append('target_column', targetCol);
    formData.append('metrics', JSON.stringify(['accuracy', 'f1']));

    try {
      const res = await fetch(`${API_BASE}/pipelines`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      
      // start execution
      await fetch(`${API_BASE}/pipelines/${data.run_id}/start?inject_failure=${injectFailure}`, {
        method: 'POST',
      });
      
      onStart(data.run_id);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto animation-fade-in mt-10">
      <div className="bg-[#1e293b]/60 backdrop-blur-xl border border-slate-700/50 p-8 rounded-2xl shadow-2xl">
        <h2 className="text-3xl font-semibold mb-6 flex items-center gap-3">
          <Settings2 className="w-8 h-8 text-indigo-400" />
          Configure ML Pipeline
        </h2>
        
        <form onSubmit={handleStart} className="space-y-6">
          {/* File Upload */}
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Dataset (CSV)</label>
            <div className="relative">
              <input 
                type="file" 
                accept=".csv"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="hidden" 
                id="file-upload" 
                required
              />
              <label 
                htmlFor="file-upload" 
                className="flex items-center justify-center gap-2 w-full px-4 py-8 bg-slate-900/50 border-2 border-dashed border-slate-700 hover:border-indigo-500 hover:bg-slate-800/50 rounded-xl cursor-pointer transition-all duration-200"
              >
                <UploadCloud className="w-6 h-6 text-slate-400" />
                <span className="text-slate-300 font-medium">
                  {file ? file.name : 'Click to upload or drag and drop'}
                </span>
              </label>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Algorithm</label>
              <select 
                value={algorithm} 
                onChange={e => setAlgorithm(e.target.value)}
                className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
              >
                <optgroup label="Classification">
                  <option value="random_forest">Random Forest</option>
                  <option value="logistic_regression">Logistic Regression</option>
                  <option value="xgboost">XGBoost</option>
                  <option value="svm">Support Vector Machine (SVM)</option>
                  <option value="gradient_boosting">Gradient Boosting</option>
                  <option value="knn">K-Nearest Neighbors (KNN)</option>
                  <option value="decision_tree">Decision Tree</option>
                </optgroup>
                <optgroup label="Regression">
                  <option value="linear_regression">Linear Regression</option>
                  <option value="ridge_regression">Ridge Regression</option>
                  <option value="lasso_regression">Lasso Regression</option>
                  <option value="random_forest_regressor">Random Forest Regressor</option>
                  <option value="xgboost_regressor">XGBoost Regressor</option>
                </optgroup>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Target Column</label>
              <input 
                type="text" 
                value={targetCol}
                onChange={e => setTargetCol(e.target.value)}
                className="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
                required
              />
            </div>
          </div>

          <div>
             <label className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl cursor-pointer">
                <input 
                  type="checkbox" 
                  checked={injectFailure}
                  onChange={e => setInjectFailure(e.target.checked)}
                  className="w-5 h-5 rounded border-slate-700 text-indigo-500 focus:ring-indigo-500/50 bg-slate-900/50"
                />
                <div>
                  <div className="text-rose-400 font-medium">Inject Mock Failure (Demo Mode)</div>
                  <div className="text-sm text-rose-500/80">Randomly injects OOM, Schema Drift, or Training Divergence so the self-healing agents can trigger.</div>
                </div>
             </label>
          </div>

          <button 
            type="submit" 
            disabled={isLoading || !file}
            className="w-full mt-4 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white py-3.5 px-4 rounded-xl font-medium transition-all duration-200 shadow-lg shadow-indigo-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <Play className="w-5 h-5" />
            )}
            Start Pipeline Run
          </button>
        </form>
      </div>
    </div>
  );
}
