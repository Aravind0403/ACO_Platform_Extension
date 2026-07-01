"""
extender/main.py
─────────────────
Kubernetes Scheduler Extender — wraps the ACO engine.

How this fits into Kubernetes
──────────────────────────────
When a pod needs to be scheduled, K8s default scheduler:
  1. Filters nodes to those that can run the pod (feasibility)
  2. Scores the remaining nodes (preference)
  3. Picks the highest-scoring node and binds the pod

A scheduler extender plugs into steps 1 and 2 via HTTP:
  POST /filter      → we veto nodes that fail our own feasibility check
  POST /prioritize  → we score nodes using ACO; K8s merges with its own scores

We don't own the final bind — K8s does. That keeps our extender stateless
and safe; if we crash, K8s falls back to its own scoring.

ACO integration
────────────────
We import the ACO engine and cost engine directly from core/.
For each scheduling call we:
  1. Parse the pod's resource requests
  2. Adapt K8s NodeInfo → our NodeState objects
  3. Run aco_schedule() → get a placement decision
  4. Convert the ACO node scores to K8s HostPriority (0–10)

The pheromone matrix persists in memory across calls (stateful within
one extender process). Across restarts pheromone resets — for a production
system you'd persist this to Redis or a file, same as the research API does.
"""

from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

# ── Path setup — import core/ modules ────────────────────────────────────────
# The extender/ directory sits next to core/ in the repo root.
# We add the repo root to sys.path so imports work regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "core"))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
)

# Import directly from modules — avoids triggering the control_plane __init__.py
# which eagerly imports WorkloadPredictor (torch-heavy). We don't need the
# predictor in the extender; predictions will be wired in during Phase 2.
import importlib.util, sys

def _import_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_models_path = _REPO_ROOT / "core" / "orchestrator" / "shared" / "models.py"
_models = _import_module("orchestrator.shared.models", str(_models_path))

ComputeNode      = _models.ComputeNode
NodeArch         = _models.NodeArch
NodeCostProfile  = _models.NodeCostProfile
InstanceType     = _models.InstanceType
JobRequest       = _models.JobRequest
ResourceRequest  = _models.ResourceRequest
WorkloadType     = _models.WorkloadType

_ce_path = _REPO_ROOT / "core" / "orchestrator" / "control_plane" / "cost_engine.py"
_ce_mod  = _import_module("orchestrator.control_plane.cost_engine", str(_ce_path))
CostEngine = _ce_mod.CostEngine
from extender.k8s_models import (
    ExtenderArgs, ExtenderFilterResult, HostPriority, NodeList, NodeInfo,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Prometheus metrics ────────────────────────────────────────────────────────
# These are scraped by Prometheus every 15s and shown in Grafana.

# How many scheduling calls we've handled total
scheduling_requests_total = Counter(
    "aco_scheduling_requests_total",
    "Total number of scheduling requests handled by the extender",
    ["endpoint"],  # 'filter' or 'prioritize'
)

# How long each prioritize call takes (ACO scoring is the interesting part)
scheduling_latency_seconds = Histogram(
    "aco_scheduling_latency_seconds",
    "Scheduling latency per request",
    ["endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# How many nodes were filtered out vs passed through
nodes_filtered_total = Counter(
    "aco_nodes_filtered_total",
    "Nodes vetoed by the filter endpoint (insufficient resources)",
)

nodes_passed_filter_total = Counter(
    "aco_nodes_passed_filter_total",
    "Nodes passed through filter to prioritize",
)

# Per-node pheromone level — lets Grafana show the heatmap
# Label: node_name
node_pheromone_level = Gauge(
    "aco_node_pheromone_level",
    "Current pheromone level for each node (ACO learning state)",
    ["node"],
)

# Which node was selected (won the prioritize round)
node_selected_total = Counter(
    "aco_node_selected_total",
    "Number of times each node was selected as best by the ACO scorer",
    ["node"],
)

# Cost of the winning node per scheduling call
scheduling_cost_usd = Histogram(
    "aco_scheduling_cost_usd_per_hour",
    "Cost per hour of the node selected for each job",
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
)

app = FastAPI(
    title="ACO Scheduler Extender",
    description="Kubernetes scheduler extender powered by Ant Colony Optimisation.",
    version="1.0.0",
)

cost_engine = CostEngine()

# Pheromone state — persists across calls within one process.
# Keyed by tenant_id so each tenant gets independent ACO learning.
# Dict[tenant_id, Dict[node_name, float]]
_pheromone: Dict[str, Dict[str, float]] = {}
_EVAPORATION = 0.05   # ρ — how much pheromone decays each call
_DEPOSIT     = 1.0    # Q — reward deposited on chosen node

def _tenant_pheromone(tenant_id: str) -> Dict[str, float]:
    """Return (and lazily create) the pheromone table for a tenant."""
    if tenant_id not in _pheromone:
        _pheromone[tenant_id] = {}
    return _pheromone[tenant_id]


# ── Resource parsing helpers ──────────────────────────────────────────────────

def _parse_cpu(cpu_str: Optional[str]) -> float:
    """Convert K8s CPU string to float cores.  '500m' → 0.5,  '2' → 2.0"""
    if not cpu_str:
        return 0.0
    if cpu_str.endswith("m"):
        return int(cpu_str[:-1]) / 1000.0
    return float(cpu_str)


def _parse_memory_gb(mem_str: Optional[str]) -> float:
    """Convert K8s memory string to float GB.  '512Mi' → 0.5,  '4Gi' → 4.0"""
    if not mem_str:
        return 0.0
    mem_str = mem_str.strip()
    units = {
        "Ki": 1 / (1024 ** 2),
        "Mi": 1 / 1024,
        "Gi": 1.0,
        "Ti": 1024.0,
        "K":  1 / (1000 ** 2),
        "M":  1 / 1000,
        "G":  1.0,
    }
    for suffix, factor in units.items():
        if mem_str.endswith(suffix):
            return float(mem_str[: -len(suffix)]) * factor
    # Plain bytes
    return float(mem_str) / (1024 ** 3)


def _pod_resources(args: ExtenderArgs) -> tuple[float, float, bool]:
    """Extract (cpu_cores, memory_gb, gpu_required) from the pod spec."""
    cpu_total = 0.0
    mem_total = 0.0
    gpu_required = False

    containers = []
    if args.Pod.spec and args.Pod.spec.containers:
        containers = args.Pod.spec.containers

    for container in containers:
        resources = container.get("resources", {})
        requests = resources.get("requests", {})
        cpu_total += _parse_cpu(requests.get("cpu"))
        mem_total += _parse_memory_gb(requests.get("memory"))
        limits = resources.get("limits", {})
        # Check for GPU request (nvidia.com/gpu or amd.com/gpu)
        if limits.get("nvidia.com/gpu") or limits.get("amd.com/gpu"):
            gpu_required = True

    return cpu_total, mem_total, gpu_required


def _node_to_state(node_info: NodeInfo) -> NodeState:
    """
    Adapt a K8s NodeInfo → our NodeState model.

    We read node labels to infer GPU type and instance tier.
    Labels we respect (set these in your minikube / GKE node pool):
      aco/gpu-type: v100 | a10 | t4 | p100 | none
      aco/instance-type: on_demand | spot
      aco/cost-per-hour: float string, e.g. "0.45"
    """
    labels = node_info.labels or {}

    # CPU / memory from allocatable (what K8s says is available for pods)
    alloc = node_info.allocatable
    total_cpu = _parse_cpu(alloc.cpu if alloc else None) or 4.0
    total_mem = _parse_memory_gb(alloc.memory if alloc else None) or 8.0

    # GPU inventory from labels
    gpu_type = labels.get("aco/gpu-type", "none").lower()
    gpu_inventory: Dict[str, int] = {}
    if gpu_type != "none":
        gpu_count = int(labels.get("aco/gpu-count", "1"))
        gpu_inventory[gpu_type] = gpu_count

    # Instance type — default on-demand
    pricing_str = labels.get("aco/instance-type", "on_demand")

    # Cost — default to a sensible mid-tier on-demand price
    cost_per_hour = float(labels.get("aco/cost-per-hour", "0.50"))

    # Architecture — default x86
    arch_str = labels.get("aco/arch", "x86_64")
    arch = NodeArch.ARM64 if "arm" in arch_str.lower() else NodeArch.X86_64

    instance_type = InstanceType.SPOT if pricing_str == "spot" else InstanceType.ON_DEMAND
    cost_profile = NodeCostProfile(
        instance_type=instance_type,
        cost_per_hour_usd=cost_per_hour,
        interruption_prob=0.1 if pricing_str == "spot" else 0.0,
    )

    return ComputeNode(
        node_id=node_info.name,
        arch=arch,
        total_cpu_cores=total_cpu,
        total_memory_gb=total_mem,
        gpu_inventory=gpu_inventory,
        cost_profile=cost_profile,
    )


def _build_job_request(cpu: float, mem: float, gpu_required: bool, workload_type: str = "batch") -> JobRequest:
    """Build a JobRequest from extracted pod resource requirements."""
    wt_map = {
        "latency-critical": WorkloadType.LATENCY_CRITICAL,
        "stream-processing": WorkloadType.STREAM,
    }
    wt = wt_map.get(workload_type, WorkloadType.BATCH)

    return JobRequest(
        job_id=str(uuid.uuid4()),
        workload_type=wt,
        resources=ResourceRequest(
            cpu_cores_min=max(cpu, 0.1),
            memory_gb_min=max(mem, 0.1),
            gpu_required=gpu_required,
            gpu_count=1 if gpu_required else 1,  # model requires >= 1; irrelevant when gpu_required=False
        ),
        priority=90 if wt == WorkloadType.LATENCY_CRITICAL else 50,
    )


def _evaporate_pheromone(table: Dict[str, float], node_names: List[str]) -> None:
    """Apply pheromone evaporation across all known nodes for one tenant."""
    for name in node_names:
        table[name] = table.get(name, 1.0) * (1 - _EVAPORATION)
        table[name] = max(table[name], 0.01)  # floor


def _deposit_pheromone(table: Dict[str, float], chosen_node: str, score: float) -> None:
    """Reward the chosen node — proportional to how good a choice it was."""
    deposit = _DEPOSIT / max(score, 0.01)
    table[chosen_node] = table.get(chosen_node, 1.0) + deposit


# ── Filter endpoint ───────────────────────────────────────────────────────────

@app.post("/filter", response_model=ExtenderFilterResult)
async def filter_nodes(args: ExtenderArgs) -> ExtenderFilterResult:
    """
    Filter: remove nodes that cannot fit the pod's resource requests.

    This is our hard gate — if a node doesn't have enough CPU or memory
    for the pod, we veto it here. K8s respects our veto.

    We intentionally keep this simple: pure feasibility. The ACO intelligence
    goes into /prioritize, not /filter.
    """
    cpu, mem, gpu_required = _pod_resources(args)
    logger.info("filter: pod needs %.2f CPU, %.2f GB mem, GPU=%s", cpu, mem, gpu_required)

    viable: List[NodeInfo] = []
    failed: Dict[str, str] = {}

    nodes = args.Nodes.items if args.Nodes else []

    for node_info in nodes:
        alloc = node_info.allocatable
        node_cpu = _parse_cpu(alloc.cpu if alloc else None) or 4.0
        node_mem = _parse_memory_gb(alloc.memory if alloc else None) or 8.0
        labels = node_info.labels or {}

        # Basic resource feasibility
        if node_cpu < cpu:
            failed[node_info.name] = f"insufficient CPU: has {node_cpu:.1f}, needs {cpu:.1f}"
            continue
        if node_mem < mem:
            failed[node_info.name] = f"insufficient memory: has {node_mem:.1f}GB, needs {mem:.1f}GB"
            continue

        # GPU feasibility
        if gpu_required:
            gpu_type = labels.get("aco/gpu-type", "none")
            if gpu_type == "none":
                failed[node_info.name] = "pod requires GPU but node has none"
                continue

        viable.append(node_info)

    scheduling_requests_total.labels(endpoint="filter").inc()
    nodes_filtered_total.inc(len(failed))
    nodes_passed_filter_total.inc(len(viable))
    logger.info("filter: %d viable, %d rejected", len(viable), len(failed))
    return ExtenderFilterResult(
        Nodes=NodeList(items=viable),
        FailedNodes=failed,
    )


# ── Prioritize endpoint ───────────────────────────────────────────────────────

@app.post("/prioritize")
async def prioritize_nodes(args: ExtenderArgs) -> list:
    """
    Prioritize: score each candidate node using the ACO cost engine.

    For each node we compute CostEngine.score_node(job, node, prediction=None).
    That gives us a composite score in (0, 1] combining:
      - Reliability  (spot interruption risk)
      - Cost efficiency
      - SLA headroom (cpu utilisation)
      - Spike prediction (if available — None here, added in Phase 2)

    We then normalise to 0–10 (what K8s expects) and update pheromone.

    The pheromone feedback means: nodes that score well now will be visited
    more often in future calls. This is the ACO learning loop.
    """
    t0 = time.perf_counter()
    scheduling_requests_total.labels(endpoint="prioritize").inc()

    cpu, mem, gpu_required = _pod_resources(args)
    nodes = args.Nodes.items if args.Nodes else []

    if not nodes:
        return []

    # Get workload type + tenant from pod labels
    pod_labels = args.Pod.metadata.get("labels", {}) if args.Pod.metadata else {}
    workload_hint = pod_labels.get("aco/workload-type", "batch")
    tenant_id = pod_labels.get("aco/tenant", "default")

    job_request = _build_job_request(cpu, mem, gpu_required, workload_hint)

    node_names = [n.name for n in nodes]
    ph = _tenant_pheromone(tenant_id)
    _evaporate_pheromone(ph, node_names)

    # Score each node
    raw_scores: Dict[str, float] = {}
    for node_info in nodes:
        node_state = _node_to_state(node_info)
        try:
            score = cost_engine.score_node(job_request, node_state, prediction=None)
        except Exception as e:
            logger.warning("cost_engine.score_node failed for %s: %s", node_info.name, e)
            score = 0.1

        # Multiply by pheromone — ACO learning loop (per-tenant)
        pheromone = ph.get(node_info.name, 1.0)
        raw_scores[node_info.name] = score * pheromone

    # Normalise to 0–10 integer scores
    max_score = max(raw_scores.values()) if raw_scores else 1.0
    priorities = []
    best_node, best_raw = max(raw_scores.items(), key=lambda x: x[1])

    for name, raw in raw_scores.items():
        k8s_score = int(round((raw / max(max_score, 1e-9)) * 10))
        priorities.append({"Host": name, "Score": k8s_score})

    # Deposit pheromone on the top-scoring node (per-tenant)
    _deposit_pheromone(ph, best_node, best_raw)

    # Update Prometheus metrics
    node_selected_total.labels(node=best_node).inc()
    for name, val in ph.items():
        node_pheromone_level.labels(node=name).set(val)

    # Record cost of selected node
    best_node_info = next((n for n in nodes if n.name == best_node), None)
    if best_node_info:
        labels = best_node_info.labels or {}
        cost = float(labels.get("aco/cost-per-hour", "0.50"))
        scheduling_cost_usd.observe(cost)

    elapsed = time.perf_counter() - t0
    scheduling_latency_seconds.labels(endpoint="prioritize").observe(elapsed)

    logger.info(
        "prioritize: %d nodes scored — best=%s (raw=%.4f, pheromone=%.4f, latency=%.3fms)",
        len(priorities), best_node, best_raw, _pheromone.get(best_node, 1.0), elapsed * 1000,
    )
    return priorities


# ── Prometheus metrics endpoint ───────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """
    Prometheus scrape endpoint. Expose all registered metrics in text format.

    Scraped by Prometheus every 15s (configured in observability/prometheus.yml).
    Grafana queries Prometheus to render the dashboard panels.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── Health check ──────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(tenant: str = None):
    """
    Reset pheromone state so the next demo run starts fresh.

    - POST /reset          → clears all tenants
    - POST /reset?tenant=X → clears only tenant X

    Grafana history stays intact (Prometheus keeps its own TSDB).
    This only resets the in-memory ACO learning state in the extender.
    """
    if tenant:
        removed = _pheromone.pop(tenant, {})
        return {"reset": "tenant", "tenant": tenant, "nodes_cleared": len(removed)}
    else:
        counts = {t: len(p) for t, p in _pheromone.items()}
        _pheromone.clear()
        return {"reset": "all", "tenants_cleared": counts}


@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "tenants": list(_pheromone.keys()),
        "pheromone_nodes_per_tenant": {t: len(p) for t, p in _pheromone.items()},
    }
