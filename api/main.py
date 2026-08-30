import asyncio
import json
import uuid
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

from orchestrator.pipeline_runner import PipelineOrchestrator
from api.database import get_connection

app = FastAPI(title="Self-Healing ML Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for active orchestrators and websocket connections
active_runs: Dict[str, PipelineOrchestrator] = {}
run_websockets: Dict[str, List[WebSocket]] = {}

class PipelineConfig(BaseModel):
    algorithm: str
    target_column: str
    metrics: list[str]

@app.post("/pipelines")
async def create_pipeline(
    file: UploadFile = File(...),
    algorithm: str = Form(...),
    target_column: str = Form(...),
    metrics: str = Form(...) # JSON string of list
):
    run_id = str(uuid.uuid4())
    config = {
        "algorithm": algorithm,
        "target_column": target_column,
        "metrics": json.loads(metrics),
        "dataset_name": file.filename
    }
    
    os.makedirs("datasets", exist_ok=True)
    with open(f"datasets/{run_id}.csv", "wb") as f:
        f.write(file.file.read())
    
    # Save to DB (initialized state)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO pipeline_runs (run_id, dataset_name, status, config)
            VALUES (?, ?, ?, ?)
        ''', (run_id, file.filename, "initialized", json.dumps(config)))
        conn.commit()
        
    return {"run_id": run_id, "config": config}

async def emit_to_websockets(run_id: str, event_type: str, data: Dict[str, Any]):
    if run_id in run_websockets:
        message = {"type": event_type, "data": data}
        for ws in run_websockets[run_id]:
            try:
                await ws.send_json(message)
            except Exception:
                pass
                
    # Update DB status if completed/failed
    if event_type == "run_completed" or event_type == "run_error":
        status = data.get("status", "error")
        failures = active_runs[run_id].failures if run_id in active_runs else {}
        
        # Save final metrics and samples to the failures JSON so it persists for history view
        if "metrics" in data:
            failures["final_metrics"] = data["metrics"]
        if "samples" in data:
            failures["final_samples"] = data["samples"]
        if "target_col" in data:
            failures["final_target_col"] = data["target_col"]
            
        recovery_used = None
        if "recovery_action" in failures:
            recovery_used = failures["recovery_action"].get("strategy_name")
            
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE pipeline_runs 
                SET status = ?, end_time = CURRENT_TIMESTAMP, failures = ?, recovery_used = ?
                WHERE run_id = ?
            ''', (status, json.dumps(failures), recovery_used, run_id))
            conn.commit()

@app.post("/pipelines/{run_id}/start")
async def start_pipeline(run_id: str, background_tasks: BackgroundTasks, inject_failure: bool = False):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT config FROM pipeline_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            return {"error": "Pipeline run not found"}
        config = json.loads(row["config"])
        
    async def run_task():
        async def cb(event_type, data):
            await emit_to_websockets(run_id, event_type, data)
        
        orchestrator = PipelineOrchestrator(run_id, cb)
        active_runs[run_id] = orchestrator
        
        with get_connection() as conn:
            conn.cursor().execute("UPDATE pipeline_runs SET status = 'running' WHERE run_id = ?", (run_id,))
            conn.commit()
            
        await orchestrator.run_pipeline(config, inject_failure=inject_failure)
        
    background_tasks.add_task(run_task)
    return {"message": "Pipeline started"}

@app.get("/pipelines/{run_id}")
async def get_pipeline_status(run_id: str):
    if run_id in active_runs:
        orchestrator = active_runs[run_id]
        return {
            "run_id": run_id,
            "status": orchestrator.status,
            "history": [s.model_dump(mode='json') for s in orchestrator.pipeline_history]
        }
        
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if not row:
            return {"error": "Pipeline run not found"}
        return {
            "run_id": row["run_id"],
            "status": row["status"],
            "dataset_name": row["dataset_name"],
            "config": json.loads(row["config"]) if row["config"] else {}
        }

@app.get("/pipelines/{run_id}/failures")
async def get_pipeline_failures(run_id: str):
    if run_id in active_runs:
        return active_runs[run_id].failures
        
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT failures FROM pipeline_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row and row["failures"]:
            return json.loads(row["failures"])
        return {}

@app.get("/pipelines/{run_id}/model")
async def download_model(run_id: str):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{run_id}.pkl"
    if not os.path.exists(model_path):
        clf = RandomForestClassifier(n_estimators=10)
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
    return FileResponse(model_path, filename=f"model_{run_id}.pkl", media_type="application/octet-stream")

@app.get("/pipelines/{run_id}/dataset")
async def download_dataset(run_id: str):
    import pandas as pd
    os.makedirs("datasets", exist_ok=True)
    dataset_path = f"datasets/{run_id}_fixed.csv"
    orig_path = f"datasets/{run_id}.csv"
    
    if not os.path.exists(dataset_path):
        if os.path.exists(orig_path):
            df = pd.read_csv(orig_path)
            # Apply mock data quality fixes (forward fill, drop remaining NaNs)
            df = df.ffill().dropna()
            df.to_csv(dataset_path, index=False)
        else:
            # Fallback to dummy dataset if original is missing
            df = pd.DataFrame({
                "age": [25, 30, 22, 45, 34],
                "income": [50000.0, 60000.0, 45000.0, 120000.0, 85000.0],
                "label": [0, 1, 0, 1, 1]
            })
            df.to_csv(dataset_path, index=False)
            
    return FileResponse(dataset_path, filename=f"fixed_dataset_{run_id}.csv", media_type="text/csv")

@app.get("/pipelines")
async def list_pipelines():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT run_id, dataset_name, status, start_time, end_time, recovery_used FROM pipeline_runs ORDER BY start_time DESC")
        return [dict(row) for row in cursor.fetchall()]

@app.get("/knowledge")
async def get_knowledge():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM knowledge_entries ORDER BY timestamp DESC")
        return [dict(row) for row in cursor.fetchall()]

@app.websocket("/pipelines/{run_id}/live")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await websocket.accept()
    if run_id not in run_websockets:
        run_websockets[run_id] = []
    run_websockets[run_id].append(websocket)
    
    # If orchestrator is already running, emit current state
    if run_id in active_runs:
        orch = active_runs[run_id]
        await websocket.send_json({"type": "current_status", "data": {"status": orch.status}})
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        run_websockets[run_id].remove(websocket)
        if not run_websockets[run_id]:
            del run_websockets[run_id]

# Mount the static React build for production monolithic deployment
# We wrap this in a check to ensure the directory exists (it won't exist if npm run build hasn't run yet)
if os.path.isdir("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

    # Fallback route for React Router client-side routing
    @app.exception_handler(404)
    async def custom_404_handler(request, exc):
        if request.url.path.startswith("/api/") or request.url.path.startswith("/pipelines/"):
            # Return standard 404 for actual API endpoints
            return {"detail": "Not Found"}
        # Return React app for all other unknown routes
        return FileResponse('frontend/dist/index.html')
