"""
api/main.py
───────────
Phase 9: FastAPI REST interface for the ACO Adaptive Compute Scheduler.

Run with:
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000 to see the live dashboard.

Architecture
─────────────
- One global OrchestratorService shared across all requests (in-memory V2).
- One global TelemetryCollector driven by a background asyncio task.
- NodeAgents are created per-job (fire-and-forget via asyncio.create_task).
- Trace adapter is swappable at runtime via POST /upload-trace.

Upgrade path to V3
───────────────────
- Replace the global svc instance with a database-backed store.
- Replace asyncio.create_task(agent.execute_job()) with a real agent
  connecting over HTTP (NodeAgent V3 uses httpx).
- Add authentication middleware.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from orchestrator.control_plane.orchestration_service import OrchestratorService
from orchestrator.data_plane.agent import NodeAgent
from orchestrator.telemetry.collector import TelemetryCollector
from orchestrator.telemetry.trace_adapter import AlibabaMachineTraceAdapter, BorgTraceAdapter

logger = logging.getLogger(__name__)

# ── Global state ─────────────────────────────────────────────────────────────

svc = OrchestratorService()
collector = TelemetryCollector(svc)

# Active NodeAgents — keyed by node_id.  Created once and kept running.
_agents: Dict[str, NodeAgent] = {}

# Interval (seconds) between telemetry ticks driven by the background loop.
_TELEMETRY_INTERVAL_S: float = 5.0

# Simulation state — auto-submit random jobs when running
_sim_task: Optional[asyncio.Task] = None
_sim_running: bool = False
_sim_interval_s: float = 8.0   # seconds between auto-submitted jobs

# Workload mix for simulation — (workload_type, cpu, mem, gpu_required, priority)
_SIM_WORKLOADS = [
    ("batch",            4.0,  8.0,  False, 30),
    ("batch",            8.0, 16.0,  False, 50),
    ("batch",            2.0,  4.0,  False, 20),
    ("latency-critical", 2.0,  4.0,  False, 90),
    ("latency-critical", 1.0,  2.0,  False, 95),
    ("stream-processing",2.0,  8.0,  False, 60),
    ("batch",            4.0, 16.0,  True,  40),   # GPU job
    ("stream-processing",1.0,  2.0,  False, 70),
]


# ── Lifespan: background tasks ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Start background telemetry loop and per-node NodeAgents on startup.
    Cancel them gracefully on shutdown.
    """
    # Restore pheromone history from last run (no-op if snapshot doesn't exist yet)
    svc.load_pheromone_snapshot()

    # Create one NodeAgent per cluster node (heartbeat only — jobs are fire-and-forget)
    for node_id in svc.node_state:
        agent = NodeAgent(node_id, svc)
        _agents[node_id] = agent
        await agent.start()

    # Background telemetry tick
    telemetry_task = asyncio.create_task(_telemetry_loop())
    logger.info("ACO Scheduler API started — %d nodes, telemetry loop running", len(svc.node_state))

    yield  # ← application runs here

    # Graceful shutdown
    global _sim_running
    _sim_running = False
    if _sim_task and not _sim_task.done():
        _sim_task.cancel()
        await asyncio.gather(_sim_task, return_exceptions=True)
    telemetry_task.cancel()
    await asyncio.gather(telemetry_task, return_exceptions=True)
    for agent in _agents.values():
        await agent.stop()
    svc.save_pheromone_snapshot()   # persist learned placement preferences for next restart
    logger.info("ACO Scheduler API shutdown complete")


async def _telemetry_loop() -> None:
    """Drive TelemetryCollector.tick() every _TELEMETRY_INTERVAL_S seconds.

    tick() is wrapped in asyncio.to_thread() so that the LSTM refit (CPU-bound
    PyTorch training, triggered every 10 ticks) does not block the event loop.
    Without this, every refit stalls all incoming API requests for ~50–200ms.

    Also drains the pending job queue on each tick — retrying saturated-cluster
    placements as resources free up from completed jobs.
    """
    try:
        while True:
            await asyncio.to_thread(collector.tick)
            # Drain queue: retry any jobs that were held due to cluster saturation.
            # Run in thread to avoid blocking event loop if many jobs re-schedule.
            outcomes = await asyncio.to_thread(svc.drain_pending_queue)
            if outcomes:
                scheduled = sum(1 for o in outcomes if o.get("status") == "SCHEDULED")
                expired   = sum(1 for o in outcomes if o.get("status") == "EXPIRED")
                logger.info("Queue drain: %d scheduled, %d expired", scheduled, expired)
            await asyncio.sleep(_TELEMETRY_INTERVAL_S)
    except asyncio.CancelledError:
        pass


async def _simulation_loop() -> None:
    """Auto-submit random jobs from _SIM_WORKLOADS on a fixed interval."""
    import random as _rnd
    try:
        while _sim_running:
            wt, cpu, mem, gpu, pri = _rnd.choice(_SIM_WORKLOADS)
            t0 = time.perf_counter()
            request_data = {
                "workload_type": wt,
                "resources": {
                    "cpu_cores_min": cpu,
                    "memory_gb_min": mem,
                    "gpu_required": gpu,
                    "gpu_count": 1,
                },
                "priority": pri,
                "preemptible": wt == "batch" and _rnd.random() < 0.3,
            }
            latency_ms = (time.perf_counter() - t0) * 1000.0
            result = await asyncio.to_thread(svc.submit_job, request_data, latency_ms)
            if result["status"] == "SCHEDULED":
                job_id = result["job_id"]
                node_id = result["node_id"]
                job_ex = svc.active_jobs.get(job_id)
                if job_ex and node_id in _agents:
                    asyncio.create_task(
                        _agents[node_id].execute_job(job_ex),
                        name=f"sim:{job_id}",
                    )
                logger.debug("Sim submitted %s → %s on %s", job_id, wt, node_id)
            await asyncio.sleep(_sim_interval_s)
    except asyncio.CancelledError:
        pass


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ACO Adaptive Compute Scheduler",
    description=(
        "Predictive, intent-aware job scheduler using Ant Colony Optimisation. "
        "Combines LSTM spike forecasting, cost-aware placement, and real cluster "
        "telemetry replay to deliver sub-10ms scheduling on heterogeneous compute."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class JobSubmitRequest(BaseModel):
    workload_type: str = "batch"
    cpu_cores_min: float = 4.0
    memory_gb_min: float = 8.0
    gpu_required: bool = False
    gpu_count: int = 1
    priority: int = 50
    preemptible: bool = False
    arch_required: Optional[str] = None
    cost_ceiling_usd: Optional[float] = None
    deadline_epoch: Optional[float] = None
    latency_p99_ms: Optional[float] = None


# ── Job endpoints ──────────────────────────────────────────────────────────────

@app.post("/jobs", summary="Submit a job to the scheduler")
async def submit_job(request: JobSubmitRequest):
    """
    Submit a job. Returns immediately with the placement decision (SCHEDULED or REJECTED).

    The job is executed asynchronously by the assigned NodeAgent in the background.
    Monitor progress via GET /jobs/{job_id}.
    """
    t0 = time.perf_counter()

    request_data = {
        "workload_type": request.workload_type,
        "resources": {
            "cpu_cores_min": request.cpu_cores_min,
            "memory_gb_min": request.memory_gb_min,
            "gpu_required": request.gpu_required,
            "gpu_count": request.gpu_count,
        },
        "priority": request.priority,
        "preemptible": request.preemptible,
    }
    if request.arch_required:
        request_data["arch_required"] = request.arch_required
    if request.cost_ceiling_usd is not None:
        request_data["cost_ceiling_usd"] = request.cost_ceiling_usd
    if request.deadline_epoch is not None:
        request_data["deadline_epoch"] = request.deadline_epoch
    if request.latency_p99_ms is not None:
        request_data["latency_p99_ms"] = request.latency_p99_ms

    latency_ms = (time.perf_counter() - t0) * 1000.0
    result = svc.submit_job(request_data, scheduling_latency_ms=latency_ms)

    if result["status"] == "SCHEDULED":
        job_id = result["job_id"]
        node_id = result["node_id"]
        job_ex = svc.active_jobs.get(job_id)
        if job_ex and node_id in _agents:
            # Fire-and-forget: execute the job on the assigned node
            asyncio.create_task(
                _agents[node_id].execute_job(job_ex),
                name=f"exec:{job_id}",
            )
        return JSONResponse(status_code=202, content=result)

    if result["status"] == "REJECTED":
        return JSONResponse(status_code=422, content=result)

    return JSONResponse(status_code=500, content=result)


@app.get("/jobs", summary="List active jobs")
async def list_active_jobs():
    """Return all currently running / scheduled jobs."""
    jobs = svc.get_active_jobs()
    return {
        "count": len(jobs),
        "jobs": [_job_summary(j) for j in jobs],
    }


@app.get("/jobs/history", summary="List recently completed jobs")
async def list_completed_jobs(limit: int = 20):
    """Return the most recently completed jobs (newest first)."""
    jobs = svc.get_completed_jobs(limit=limit)
    return {
        "count": len(jobs),
        "jobs": [_job_summary(j) for j in jobs],
    }


@app.get("/jobs/{job_id}", summary="Get job status")
async def get_job(job_id: str):
    """Get the current state of a specific job (active or completed)."""
    job = svc.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return _job_summary(job)


def _job_summary(job) -> dict:
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "workload_type": job.job_request.workload_type.value,
        "assigned_node": job.assigned_node_id,
        "cpu_cores": job.job_request.resources.cpu_cores_min,
        "memory_gb": job.job_request.resources.memory_gb_min,
        "gpu_required": job.job_request.resources.gpu_required,
        "priority": job.job_request.priority,
        "submitted_at": job.submitted_at.isoformat() if job.submitted_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "scheduling_latency_ms": job.scheduling_latency_ms,
    }


# ── Node endpoints ─────────────────────────────────────────────────────────────

@app.get("/nodes", summary="List cluster nodes with live utilisation")
async def list_nodes():
    """
    Return all cluster nodes with current CPU/memory allocation and
    live telemetry (if a heartbeat has been received).
    """
    nodes = []
    for node_id, node in svc.node_state.items():
        telemetry = node.latest_telemetry
        entry: Dict[str, Any] = {
            "node_id": node_id,
            "arch": node.arch.value,
            "instance_type": node.cost_profile.instance_type.value,
            "total_cpu_cores": node.total_cpu_cores,
            "total_memory_gb": node.total_memory_gb,
            "allocated_cpu_cores": round(node.allocated_cpu_cores, 2),
            "allocated_memory_gb": round(node.allocated_memory_gb, 2),
            "available_cpu_cores": round(node.available_cpu_cores, 2),
            "available_memory_gb": round(node.available_memory_gb, 2),
            "cpu_allocation_pct": round(node.cpu_utilisation_pct, 1),
            "cost_per_hour_usd": node.cost_profile.cost_per_hour_usd,
            "gpu_inventory": node.gpu_inventory,
            "state": node.state.value,
        }
        if telemetry:
            entry["live_telemetry"] = {
                "cpu_util_pct": round(telemetry.cpu_util_pct, 1),
                "memory_util_pct": round(telemetry.memory_util_pct, 1),
                "gpu_util_pct": telemetry.gpu_util_pct,
            }
        else:
            entry["live_telemetry"] = None
        nodes.append(entry)
    return {"count": len(nodes), "nodes": nodes}


# ── Queue endpoint ─────────────────────────────────────────────────────────────

@app.get("/queue", summary="Pending job queue status")
async def get_queue():
    """
    Return the current state of the back-pressure queue.

    When the cluster is fully saturated, new jobs are held here instead of
    being silently rejected. The queue drains automatically every
    `_TELEMETRY_INTERVAL_S` seconds as resources free up.

    - `depth`: number of jobs currently waiting
    - `oldest_age_s`: age of the oldest queued job in seconds
    - `max_depth`: maximum queue capacity (default 50)
    - `ttl_s`: job expiry time-to-live in seconds (default 60)
    """
    return svc.get_queue_status()


# ── Simulation endpoints ───────────────────────────────────────────────────────

@app.post("/simulation/start", summary="Start auto-submitting random jobs")
async def start_simulation(interval_s: float = 8.0):
    """
    Start the simulation loop — random jobs are submitted every `interval_s` seconds.
    The workload mix covers batch, latency-critical, stream-processing, and GPU jobs.
    """
    global _sim_task, _sim_running, _sim_interval_s
    if _sim_running:
        return {"status": "already_running", "interval_s": _sim_interval_s}
    _sim_interval_s = max(1.0, interval_s)
    _sim_running = True
    _sim_task = asyncio.create_task(_simulation_loop(), name="simulation")
    return {"status": "started", "interval_s": _sim_interval_s}


@app.post("/simulation/stop", summary="Stop auto-submitting jobs")
async def stop_simulation():
    """Stop the simulation loop. Existing running jobs continue to completion."""
    global _sim_running
    _sim_running = False
    if _sim_task and not _sim_task.done():
        _sim_task.cancel()
        await asyncio.gather(_sim_task, return_exceptions=True)
    return {"status": "stopped"}


@app.get("/simulation/status", summary="Is the simulation loop running?")
async def simulation_status():
    return {
        "running": _sim_running,
        "interval_s": _sim_interval_s,
        "workload_types": list({w[0] for w in _SIM_WORKLOADS}),
    }


# ── Metrics endpoint ───────────────────────────────────────────────────────────

@app.get("/metrics", summary="Scheduling performance metrics")
async def get_metrics():
    """
    Return P99 scheduling latency, average utilisation, active/completed counts,
    and per-node prediction confidence.
    """
    base = svc.get_scheduling_metrics()

    # Augment with prediction availability
    predictions: Dict[str, Any] = {}
    for node_id in svc.node_state:
        pred = svc.get_prediction(node_id)
        if pred:
            predictions[node_id] = {
                "predicted_cpu_util": round(pred.predicted_cpu_util, 1),
                "spike_probability": round(pred.spike_probability, 3),
                "confidence": round(pred.confidence, 3),
            }

    base["predictions"] = predictions
    adapter = collector._trace_adapter
    if isinstance(adapter, BorgTraceAdapter):
        base["telemetry_source"] = "borg-2019"
    elif isinstance(adapter, AlibabaMachineTraceAdapter):
        base["telemetry_source"] = "alibaba-2018"
    else:
        base["telemetry_source"] = "synthetic-gaussian"
    return base


# ── Prediction endpoint ────────────────────────────────────────────────────────

@app.get("/predict/{node_id}", summary="LSTM CPU spike prediction for a node")
async def get_prediction(node_id: str):
    """
    Return the latest LSTM prediction for the given node.

    - `predicted_cpu_util`: forecast CPU % for the next 5 minutes
    - `spike_probability`: probability of a CPU spike [0.0, 1.0]
    - `confidence`: prediction confidence based on training sample count [0.0, 1.0]
    """
    if node_id not in svc.node_state:
        raise HTTPException(status_code=404, detail=f"Node {node_id!r} not found")

    pred = svc.get_prediction(node_id)
    if pred is None:
        return {
            "node_id": node_id,
            "status": "not_ready",
            "message": "Predictor not yet trained (need ≥10 telemetry samples)",
        }

    return {
        "node_id": node_id,
        "status": "ready",
        "predicted_cpu_util": round(pred.predicted_cpu_util, 2),
        "predicted_memory_util": round(pred.predicted_memory_util, 2),
        "spike_probability": round(pred.spike_probability, 4),
        "confidence": round(pred.confidence, 4),
        "forecast_horizon_min": pred.forecast_horizon_min,
    }


# ── Trace upload endpoint ──────────────────────────────────────────────────────

@app.post("/upload-trace", summary="Upload a custom cluster trace CSV")
async def upload_trace(file: UploadFile = File(...)):
    """
    Upload a custom cluster trace CSV to replace the current telemetry source.

    **Required columns:** `cpu_util_percent`, `mem_util_percent`

    Both columns must contain numeric values in the range [0, 100].
    The system will immediately start using the uploaded trace for all
    subsequent telemetry ticks.

    **Supported formats (auto-detected):**

    - **Alibaba 2018** (Zenodo record 14564935): columns `cpu_util_percent`, `mem_util_percent`
    - **Google Cluster Trace 2019 / Borg** (Kaggle): columns `time`, `average_usage`
      where `average_usage` is a dict string like `{'cpus': 0.021, 'memory': 0.014}`.

    Returns the number of timesteps loaded, detected format, and a preview.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    _MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
    content = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    import csv as csv_module
    import os
    import tempfile

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="wb") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = content.decode("utf-8", errors="replace")
        reader = csv_module.DictReader(io.StringIO(text))
        fieldnames = list(reader.fieldnames or [])

        # Auto-detect format
        is_alibaba = "cpu_util_percent" in fieldnames and "mem_util_percent" in fieldnames
        is_borg = "time" in fieldnames and "average_usage" in fieldnames

        if not is_alibaba and not is_borg:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Unrecognised CSV format. "
                    "Alibaba format needs: 'cpu_util_percent', 'mem_util_percent'. "
                    "Borg/Google format needs: 'time', 'average_usage'. "
                    f"Found columns: {fieldnames[:10]}{'...' if len(fieldnames) > 10 else ''}"
                ),
            )

        rows = list(reader)
        if not rows:
            raise HTTPException(status_code=422, detail="CSV has headers but no data rows")

        if is_alibaba:
            new_adapter = AlibabaMachineTraceAdapter(tmp_path)
            detected_format = "alibaba-2018"
            preview = {
                "cpu_util_percent": rows[0].get("cpu_util_percent"),
                "mem_util_percent": rows[0].get("mem_util_percent"),
            }
            extra = {}
        else:
            new_adapter = BorgTraceAdapter(tmp_path)
            detected_format = "borg-2019"
            preview = {"average_usage": rows[0].get("average_usage", "")[:80]}
            extra = {"raw_rows_parsed": new_adapter.raw_rows}

        # Hot-swap the collector's adapter
        collector._trace_adapter = new_adapter

        return {
            "status": "ok",
            "filename": file.filename,
            "detected_format": detected_format,
            "timesteps_loaded": new_adapter.trace_length,
            "columns_detected": fieldnames[:10],
            "preview": preview,
            "message": (
                f"[{detected_format}] Trace loaded: {new_adapter.trace_length} timesteps. "
                "All future telemetry ticks will use this trace."
            ),
            **extra,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Serve the built-in real-time dashboard."""
    return HTMLResponse(content=_DASHBOARD_HTML)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ACO Scheduler Dashboard</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #252836;
    --border: #2e3248;
    --text: #e2e8f0;
    --muted: #64748b;
    --accent: #6366f1;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
    --blue: #3b82f6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; font-size: 14px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 16px 24px; display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 18px; font-weight: 700; letter-spacing: -0.3px; }
  header h1 span { color: var(--accent); }
  .badge { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; font-size: 12px; color: var(--muted); }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--green); margin-right: 6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
  main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 20px 24px; max-width: 1400px; margin: 0 auto; }
  .full { grid-column: 1 / -1; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card h2 { font-size: 13px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; }
  .stat-row { display: flex; gap: 20px; }
  .stat { flex: 1; }
  .stat .val { font-size: 32px; font-weight: 700; }
  .stat .lbl { font-size: 12px; color: var(--muted); margin-top: 2px; }
  .node-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
  .node-card { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 14px; }
  .node-id { font-weight: 600; font-size: 13px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between; }
  .arch-badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; background: var(--border); color: var(--muted); }
  .bar-wrap { margin-bottom: 8px; }
  .bar-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 3px; }
  .bar-bg { background: var(--border); border-radius: 99px; height: 6px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 99px; transition: width 0.5s ease; }
  .bar-cpu { background: var(--accent); }
  .bar-mem { background: var(--blue); }
  .bar-alloc { background: var(--amber); }
  .pred-row { display: flex; gap: 8px; margin-top: 8px; font-size: 11px; }
  .pred-chip { background: var(--border); border-radius: 4px; padding: 2px 6px; }
  .spike-high { background: rgba(239,68,68,0.2); color: var(--red); }
  .spike-med { background: rgba(245,158,11,0.2); color: var(--amber); }
  .spike-low { background: rgba(34,197,94,0.2); color: var(--green); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; color: var(--muted); font-weight: 600; padding: 6px 8px; border-bottom: 1px solid var(--border); }
  td { padding: 8px 8px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  .state-RUNNING { color: var(--green); }
  .state-COMPLETED { color: var(--muted); }
  .state-FAILED { color: var(--red); }
  .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
  input, select { width: 100%; background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; color: var(--text); font-size: 13px; outline: none; }
  input:focus, select:focus { border-color: var(--accent); }
  .btn { background: var(--accent); color: #fff; border: none; border-radius: 8px; padding: 10px 20px; font-weight: 600; cursor: pointer; font-size: 13px; margin-top: 12px; transition: opacity 0.15s; }
  .btn:hover { opacity: 0.85; }
  .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text); }
  .upload-zone { border: 2px dashed var(--border); border-radius: 8px; padding: 20px; text-align: center; cursor: pointer; transition: border-color 0.15s; }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--accent); }
  .upload-zone p { color: var(--muted); font-size: 12px; margin-top: 6px; }
  .msg { padding: 8px 12px; border-radius: 6px; font-size: 12px; margin-top: 10px; }
  .msg-ok { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); color: var(--green); }
  .msg-err { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); color: var(--red); }
  .source-chip { font-size: 11px; background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; color: var(--muted); }
  .sim-btn { border: none; border-radius: 8px; padding: 7px 16px; font-weight: 600; cursor: pointer; font-size: 12px; transition: opacity 0.15s; }
  .sim-start { background: var(--green); color: #000; }
  .sim-stop  { background: var(--red);   color: #fff; }
  .sim-btn:hover { opacity: 0.8; }
</style>
</head>
<body>
<header>
  <h1>ACO <span>Adaptive</span> Compute Scheduler</h1>
  <div style="display:flex;align-items:center;gap:12px">
    <span id="source-chip" class="source-chip">telemetry: initialising…</span>
    <button id="sim-btn" class="sim-btn sim-start" onclick="toggleSim()">▶ Start Simulation</button>
    <span class="badge"><span class="status-dot"></span>Live</span>
  </div>
</header>

<main>

<!-- Metrics row -->
<div class="card full">
  <h2>Scheduling Metrics</h2>
  <div class="stat-row">
    <div class="stat"><div class="val" id="m-active">–</div><div class="lbl">Active Jobs</div></div>
    <div class="stat"><div class="val" id="m-completed">–</div><div class="lbl">Completed</div></div>
    <div class="stat"><div class="val" id="m-p99">–</div><div class="lbl">P99 Latency (ms)</div></div>
    <div class="stat"><div class="val" id="m-avg">–</div><div class="lbl">Avg Latency (ms)</div></div>
  </div>
</div>

<!-- Node cards -->
<div class="card full">
  <h2>Cluster Nodes</h2>
  <div class="node-grid" id="node-grid"></div>
</div>

<!-- Job history -->
<div class="card full">
  <h2>Recent Jobs</h2>
  <table>
    <thead><tr>
      <th>Job ID</th><th>Type</th><th>Node</th><th>CPU</th><th>Mem (GB)</th><th>Priority</th><th>State</th><th>Latency (ms)</th>
    </tr></thead>
    <tbody id="job-table"></tbody>
  </table>
</div>

<!-- Submit job form -->
<div class="card">
  <h2>Submit a Job</h2>
  <div class="form-grid">
    <div>
      <label>Workload Type</label>
      <select id="f-type">
        <option value="batch">Batch</option>
        <option value="latency-critical">Latency Critical</option>
        <option value="stream-processing">Stream Processing</option>
      </select>
    </div>
    <div>
      <label>Priority (1–100)</label>
      <input type="number" id="f-priority" value="50" min="1" max="100">
    </div>
    <div>
      <label>CPU Cores Min</label>
      <input type="number" id="f-cpu" value="4" step="0.5" min="0.5">
    </div>
    <div>
      <label>Memory GB Min</label>
      <input type="number" id="f-mem" value="8" step="1" min="1">
    </div>
    <div>
      <label>GPU Required</label>
      <select id="f-gpu">
        <option value="false">No</option>
        <option value="true">Yes</option>
      </select>
    </div>
    <div>
      <label>Preemptible</label>
      <select id="f-preemptible">
        <option value="false">No</option>
        <option value="true">Yes</option>
      </select>
    </div>
  </div>
  <button class="btn" onclick="submitJob()">Submit Job</button>
  <div id="submit-msg" style="display:none" class="msg"></div>
</div>

<!-- Trace upload form -->
<div class="card">
  <h2>Upload Cluster Trace</h2>
  <p style="color:var(--muted);font-size:12px;margin-bottom:12px">
    Upload a CSV with <code>cpu_util_percent</code> and <code>mem_util_percent</code> columns
    to replace synthetic telemetry with your own cluster data.
    Compatible with the Alibaba 2018 cluster trace format.
  </p>
  <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
    <div style="font-size:24px">📂</div>
    <div style="font-weight:600;margin-top:4px" id="drop-label">Click or drag a CSV file here</div>
    <p>Required columns: cpu_util_percent, mem_util_percent</p>
  </div>
  <input type="file" id="file-input" accept=".csv" style="display:none" onchange="uploadTrace(this.files[0])">
  <div id="upload-msg" style="display:none" class="msg"></div>
</div>

</main>

<script>
// ── Polling ─────────────────────────────────────────────────────────────────

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

let _simRunning = false;

async function refresh() {
  try {
    const [metrics, nodes, activeJobs, historyJobs, simStatus] = await Promise.all([
      fetchJSON('/metrics'),
      fetchJSON('/nodes'),
      fetchJSON('/jobs'),
      fetchJSON('/jobs/history?limit=15'),
      fetchJSON('/simulation/status'),
    ]);
    renderMetrics(metrics);
    renderNodes(nodes.nodes, metrics.predictions);
    renderJobs(activeJobs.jobs, historyJobs.jobs);
    renderSimStatus(simStatus);
  } catch (e) {
    console.error('Refresh error:', e);
  }
}

function renderSimStatus(s) {
  _simRunning = s.running;
  const btn = document.getElementById('sim-btn');
  if (s.running) {
    btn.textContent = `⏹ Stop Simulation (every ${s.interval_s}s)`;
    btn.className = 'sim-btn sim-stop';
  } else {
    btn.textContent = '▶ Start Simulation';
    btn.className = 'sim-btn sim-start';
  }
}

async function toggleSim() {
  if (_simRunning) {
    await fetch('/simulation/stop', {method:'POST'});
  } else {
    await fetch('/simulation/start?interval_s=8', {method:'POST'});
  }
  refresh();
}

function renderMetrics(m) {
  document.getElementById('m-active').textContent = m.active_jobs;
  document.getElementById('m-completed').textContent = m.completed_jobs;
  document.getElementById('m-p99').textContent = m.scheduling_p99_ms.toFixed(2);
  document.getElementById('m-avg').textContent = m.avg_scheduling_ms.toFixed(2);
  document.getElementById('source-chip').textContent = 'telemetry: ' + (m.telemetry_source || 'unknown');
}

function renderNodes(nodes, predictions) {
  const grid = document.getElementById('node-grid');
  grid.innerHTML = nodes.map(n => {
    const t = n.live_telemetry;
    const cpuLive = t ? t.cpu_util_pct : null;
    const memLive = t ? t.memory_util_pct : null;
    const allocPct = n.total_cpu_cores > 0 ? (n.allocated_cpu_cores / n.total_cpu_cores * 100) : 0;
    const pred = predictions && predictions[n.node_id];

    const spikeClass = !pred ? '' : pred.spike_probability > 0.6 ? 'spike-high' :
      pred.spike_probability > 0.3 ? 'spike-med' : 'spike-low';

    const gpuBadge = n.gpu_inventory && Object.keys(n.gpu_inventory).length > 0
      ? ' 🎮' : '';

    const archLabel = n.arch.replace('_', ' ');

    return `<div class="node-card">
      <div class="node-id">
        <span>${n.node_id}${gpuBadge}</span>
        <span class="arch-badge">${archLabel} ${n.instance_type}</span>
      </div>
      ${cpuLive !== null ? `
      <div class="bar-wrap">
        <div class="bar-label"><span>Live CPU</span><span>${cpuLive.toFixed(1)}%</span></div>
        <div class="bar-bg"><div class="bar-fill bar-cpu" style="width:${cpuLive}%"></div></div>
      </div>
      <div class="bar-wrap">
        <div class="bar-label"><span>Live Mem</span><span>${memLive ? memLive.toFixed(1) : '–'}%</span></div>
        <div class="bar-bg"><div class="bar-fill bar-mem" style="width:${memLive || 0}%"></div></div>
      </div>` : '<div style="color:var(--muted);font-size:11px;margin-bottom:8px">Awaiting heartbeat…</div>'}
      <div class="bar-wrap">
        <div class="bar-label"><span>CPU Allocated</span><span>${allocPct.toFixed(1)}%</span></div>
        <div class="bar-bg"><div class="bar-fill bar-alloc" style="width:${allocPct}%"></div></div>
      </div>
      <div class="pred-row">
        ${pred ? `
          <span class="pred-chip">Forecast: ${pred.predicted_cpu_util.toFixed(1)}%</span>
          <span class="pred-chip ${spikeClass}">Spike: ${(pred.spike_probability * 100).toFixed(0)}%</span>
          <span class="pred-chip">Conf: ${(pred.confidence * 100).toFixed(0)}%</span>
        ` : '<span class="pred-chip" style="color:var(--muted)">No prediction yet</span>'}
      </div>
    </div>`;
  }).join('');
}

function renderJobs(active, history) {
  const all = [...active.map(j => ({...j, _active: true})), ...history.slice(0, 15)];
  const tbody = document.getElementById('job-table');
  if (all.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="color:var(--muted);text-align:center;padding:20px">No jobs yet — submit one below</td></tr>';
    return;
  }
  tbody.innerHTML = all.slice(0, 15).map(j => `<tr>
    <td style="font-family:monospace;color:var(--accent)">${j.job_id}</td>
    <td>${j.workload_type}</td>
    <td style="color:var(--muted)">${j.assigned_node || '–'}</td>
    <td>${j.cpu_cores}</td>
    <td>${j.memory_gb}</td>
    <td>${j.priority}</td>
    <td class="state-${j.state}">${j.state}</td>
    <td>${j.scheduling_latency_ms ? j.scheduling_latency_ms.toFixed(2) : '–'}</td>
  </tr>`).join('');
}

// ── Submit job ───────────────────────────────────────────────────────────────

async function submitJob() {
  const body = {
    workload_type: document.getElementById('f-type').value,
    priority: parseInt(document.getElementById('f-priority').value),
    cpu_cores_min: parseFloat(document.getElementById('f-cpu').value),
    memory_gb_min: parseFloat(document.getElementById('f-mem').value),
    gpu_required: document.getElementById('f-gpu').value === 'true',
    gpu_count: 1,
    preemptible: document.getElementById('f-preemptible').value === 'true',
  };
  const msgEl = document.getElementById('submit-msg');
  try {
    const r = await fetch('/jobs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const data = await r.json();
    msgEl.style.display = 'block';
    msgEl.className = 'msg ' + (r.ok ? 'msg-ok' : 'msg-err');
    msgEl.textContent = data.message || JSON.stringify(data);
    setTimeout(() => { msgEl.style.display = 'none'; }, 5000);
    refresh();
  } catch(e) {
    msgEl.style.display = 'block';
    msgEl.className = 'msg msg-err';
    msgEl.textContent = 'Error: ' + e.message;
  }
}

// ── Upload trace ─────────────────────────────────────────────────────────────

async function uploadTrace(file) {
  if (!file) return;
  document.getElementById('drop-label').textContent = `Uploading ${file.name}…`;
  const fd = new FormData();
  fd.append('file', file);
  const msgEl = document.getElementById('upload-msg');
  try {
    const r = await fetch('/upload-trace', { method: 'POST', body: fd });
    const data = await r.json();
    msgEl.style.display = 'block';
    msgEl.className = 'msg ' + (r.ok ? 'msg-ok' : 'msg-err');
    msgEl.textContent = r.ok
      ? `✓ ${data.message}`
      : `✗ ${data.detail || JSON.stringify(data)}`;
    document.getElementById('drop-label').textContent = r.ok
      ? `Loaded: ${file.name} (${data.timesteps_loaded} timesteps)`
      : 'Click or drag a CSV file here';
  } catch(e) {
    msgEl.style.display = 'block';
    msgEl.className = 'msg msg-err';
    msgEl.textContent = 'Upload failed: ' + e.message;
    document.getElementById('drop-label').textContent = 'Click or drag a CSV file here';
  }
}

// ── Drag-and-drop ────────────────────────────────────────────────────────────

const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) uploadTrace(f);
});

// ── Auto-refresh ─────────────────────────────────────────────────────────────

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""
