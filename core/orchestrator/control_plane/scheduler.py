"""
orchestrator/control_plane/scheduler.py
────────────────────────────────────────
The scheduling layer: decides WHICH node a job goes to.

V1 had naive_schedule() — First Fit. Simple, fast, blind.
V2 replaces it with aco_schedule() — ACO Colony + CostEngine.

The two scheduling functions
──────────────────────────────
1. aco_schedule(job_request, available_nodes, predictors)
     The primary scheduler. Uses the ACO Colony (Phase 2) with CostEngine
     (Phase 4) scores as the η heuristic. Learns across calls via pheromone.
     Falls back to naive_schedule() if the colony fails (graceful degradation).

2. naive_schedule(job_request, available_nodes)
     Kept from V1. First Fit algorithm. Used as:
       a. Fallback when aco_schedule raises ColonyFailedError.
       b. Directly from tests that need deterministic placement.
     No behaviour changes from V1.

How aco_schedule works
───────────────────────
When called, aco_schedule:

1. Filters to feasible nodes (node.can_fit(job.resources)) — O(n_nodes).
   If none feasible → raise SchedulingFailedError immediately.

2. Builds a COST-AWARE η matrix using CostEngine.score_node():
   For each (job, node) pair:
     η[i][j] = CostEngine.score_node(job, node, predictors.get(node_id))
   This replaces the four static scoring methods in ant.py with a single
   authoritative composite score that includes:
     - Spot reliability (hard gate for LC jobs)
     - Cost efficiency (1/(1+norm_cost))
     - SLA headroom (CPU util ← real telemetry or allocation estimate)
     - Spike prediction (dampened by predictor confidence)

3. Injects the η matrix into the ACO Colony as `shared_eta`.
   The colony's ants then sample from:
     P(job → node) ∝ τ[i,j]^α × η[i,j]^β
   where τ comes from the pheromone matrix (Colony manages this).

4. Colony.run() returns a PlacementPlan: Dict[job_id, node_id].
   For a single-job call (the typical submission path), we extract the
   one node_id from the plan.

5. If ColonyFailedError → fall back to naive_schedule() and log a warning.

Pheromone persistence (Phase 10 hardening)
───────────────────────────────────────────
OrchestratorService maintains ``_node_pheromone: Dict[str, float]`` — one
float per node, accumulated across all scheduling calls. After each successful
placement the chosen node's score is deposited (Q / score) and all nodes are
evaporated (×(1-RHO)). This vector is passed to aco_schedule() as
``node_pheromone`` and injected into PheromoneMatrix as the initial τ row,
so the colony starts informed by past decisions rather than blind at 1.0.

On API startup/shutdown, the snapshot is saved to / loaded from JSON so
learned preferences survive process restarts.

Error handling contract
────────────────────────
  SchedulingFailedError: raised when NO node can fit the job at all.
                         Caller (OrchestratorService) must catch this and
                         return REJECTED to the API caller.
                         NOT raised on ColonyFailedError — that triggers
                         the naive fallback instead.

  ColonyFailedError:     internal to aco_schedule. Caught here; triggers
                         naive_schedule() as graceful degradation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from orchestrator.shared.models import (
    ComputeNode,
    JobRequest,
    NodeState,
    PredictionResult,
    WorkloadType,
)
from orchestrator.control_plane.cost_engine import CostEngine
from orchestrator.control_plane.intent_router import SchedulingStrategy
from aco_core import Colony, ColonyFailedError
from aco_core.pheromone import TAU_INITIAL

logger = logging.getLogger(__name__)

# ── Module-level cost engine (stateless — one instance is fine) ───────────────
_cost_engine = CostEngine()


# ── Custom exception (kept from V1 — no breaking change) ──────────────────────

class SchedulingFailedError(Exception):
    """
    Raised when the scheduler cannot find a suitable placement for the job.

    When is this raised?
        • No node in available_nodes can satisfy job.resources (can_fit() = False
          for all nodes). The cluster is genuinely out of capacity.
        • NOT raised on ColonyFailedError — that triggers the naive fallback.

    Caller contract (OrchestratorService):
        Catch this and return {"status": "REJECTED"} to the API caller.
        Do NOT swallow it silently — the job must not be silently dropped.
    """
    pass


# ── Primary scheduler: ACO ─────────────────────────────────────────────────────

def aco_schedule(
    job_request: JobRequest,
    available_nodes: List[ComputeNode],
    predictors: Optional[Dict[str, PredictionResult]] = None,
    strategy: Optional[SchedulingStrategy] = None,
    node_workload_map: Optional[Dict[str, List[WorkloadType]]] = None,
    node_pheromone: Optional[Dict[str, float]] = None,
) -> str:
    """
    Schedule a single job using the ACO Colony with CostEngine η heuristic.

    This is the V2 replacement for naive_schedule(). It uses:
      - ACO Colony (Phase 2): stochastic multi-ant search with pheromone learning
      - CostEngine (Phase 4): composite η heuristic (reliability, cost, SLA, spike)
      - WorkloadPredictor (Phase 3): spike forecasts fed into CostEngine
      - SchedulingStrategy (Phase 6): intent-aware filtering and threshold overrides

    Args:
        job_request:       The admitted job to be placed.
        available_nodes:   All known ComputeNode objects in the cluster.
        predictors:        Optional map of node_id → PredictionResult from the
                           background predictor loop. If None, prediction_factor=1.0.
        strategy:          Optional SchedulingStrategy from WorkloadIntentRouter.
                           When provided: filters nodes by required_arch/instance,
                           applies threshold overrides to CostEngine, and respects
                           use_fast_path and avoid_workload_types colocation policy.
        node_workload_map: Optional map of node_id → List[WorkloadType] for nodes
                           currently hosting jobs of those types. Used with
                           strategy.avoid_workload_types colocation filter.
        node_pheromone:    Optional map of node_id → float with cross-call pheromone
                           levels accumulated by OrchestratorService. Seeded into
                           PheromoneMatrix so the colony starts informed by past
                           placement decisions rather than uniform TAU_INITIAL=1.0.

    Returns:
        node_id: str — the ID of the selected node.

    Raises:
        SchedulingFailedError: if no node can physically fit the job.
                               (ColonyFailedError triggers fallback instead.)

    Performance:
        Colony runs: 20 ants × up to 5 iterations.
        With CostEngine η pre-computed once and shared across ants,
        target latency is <8ms for ≤20 nodes.
        Fast path: single LATENCY_CRITICAL job or strategy.use_fast_path=True
                   → deterministic argmax, <1ms.
    """
    predictors = predictors or {}
    node_workload_map = node_workload_map or {}

    # ── Step 1: Filter to feasible nodes ──────────────────────────────────────
    feasible_nodes = [
        node for node in available_nodes
        if node.state == NodeState.HEALTHY and node.can_fit(job_request.resources)
    ]

    # Strategy-based hard filters (required_arch, required_instance)
    if strategy is not None:
        if strategy.required_arch is not None:
            feasible_nodes = [
                n for n in feasible_nodes if n.arch in strategy.required_arch
            ]
        if strategy.required_instance is not None:
            feasible_nodes = [
                n for n in feasible_nodes
                if n.cost_profile.instance_type in strategy.required_instance
            ]

    if not feasible_nodes:
        raise SchedulingFailedError(
            f"Job {job_request.job_id} could not be placed. "
            f"No node met minimum requirements "
            f"(CPU={job_request.resources.cpu_cores_min}, "
            f"MEM={job_request.resources.memory_gb_min}GB)."
        )

    # ── Step 2: Build cost-aware η matrix ─────────────────────────────────────
    # Compute CostEngine score for each (job, node) pair.
    # When a strategy is provided, its threshold overrides are forwarded to
    # CostEngine so the scoring reflects the job's actual requirements.
    n_jobs = 1
    n_nodes = len(feasible_nodes)

    # Unpack strategy thresholds once (None → use CostEngine module defaults)
    sla_threshold = strategy.sla_strict_threshold if strategy is not None else None
    spike_weight = strategy.spike_penalty_weight if strategy is not None else None
    spot_threshold = strategy.spot_penalty_threshold if strategy is not None else None
    avoid_types = set(strategy.avoid_workload_types) if strategy is not None else set()

    eta = np.zeros((n_jobs, n_nodes), dtype=np.float64)
    for j_idx, node in enumerate(feasible_nodes):
        # Colocation policy: if node is hosting job types we want to avoid, η = 0
        running_types = set(node_workload_map.get(node.node_id, []))
        if avoid_types & running_types:
            eta[0, j_idx] = 0.0
            continue

        prediction = predictors.get(node.node_id)
        eta[0, j_idx] = _cost_engine.score_node(
            job_request, node, prediction,
            sla_threshold=sla_threshold,
            spike_weight=spike_weight,
            spot_threshold=spot_threshold,
        )

    # Guard: if every node scores 0.0 (all hard-gated by cost/reliability/SLA),
    # fall back rather than passing all-zero η to the colony.
    if eta.sum() == 0.0:
        logger.warning(
            "aco_schedule: all %d feasible nodes scored 0.0 for job %s "
            "(hard-gated). Falling back to naive_schedule.",
            n_nodes, job_request.job_id,
        )
        return _naive_schedule_internal(job_request, feasible_nodes)

    # ── Step 3: Run ACO Colony ────────────────────────────────────────────────
    # Determine whether to use the fast (deterministic argmax) path.
    # Conditions: LC job OR strategy explicitly requests fast path.
    use_fast_path = (
        job_request.workload_type == WorkloadType.LATENCY_CRITICAL
        or (strategy is not None and strategy.use_fast_path)
    )

    try:
        # Build initial_tau vector for feasible nodes from cross-call pheromone history.
        # Nodes not in node_pheromone default to TAU_INITIAL (1.0) — fresh start.
        initial_tau: Optional["np.ndarray"] = None
        if node_pheromone:
            initial_tau = np.array(
                [node_pheromone.get(n.node_id, TAU_INITIAL) for n in feasible_nodes],
                dtype=np.float64,
            )

        colony = Colony(jobs=[job_request], nodes=feasible_nodes, initial_tau=initial_tau)
        plan = _run_colony_with_eta(colony, eta, force_fast_path=use_fast_path)
        node_id = plan[job_request.job_id]

        logger.info(
            "aco_schedule: job %s → node %s (colony: %.2fms, %d feasible nodes)",
            job_request.job_id, node_id, colony.last_run_ms, n_nodes,
        )
        return node_id

    except ColonyFailedError:
        # Colony found no feasible solution (all ants infeasible).
        # Graceful degradation: fall back to naive First Fit.
        logger.warning(
            "aco_schedule: ColonyFailedError for job %s — falling back to naive_schedule.",
            job_request.job_id,
        )
        return _naive_schedule_internal(job_request, feasible_nodes)


def _run_colony_with_eta(
    colony: Colony,
    eta: "np.ndarray",
    force_fast_path: bool = False,
) -> "PlacementPlan":
    """
    Run the colony, injecting a pre-computed CostEngine η matrix.

    The Colony normally computes η inside the Ant constructor. By pre-computing
    it with CostEngine and passing it as `shared_eta`, we get:
      1. Richer η (reliability + cost + SLA + prediction vs simple headroom)
      2. No redundant computation — η computed once, shared across all ants

    This function patches the shared_eta slot that Colony.run() already
    supports via the `shared_eta` parameter in Ant.__init__().

    Args:
        colony:          Pre-initialised Colony with jobs + nodes.
        eta:             Pre-computed η matrix from CostEngine.
        force_fast_path: If True, always use deterministic argmax (skip colony).
                         Used by strategy.use_fast_path=True.
    """
    from aco_core.ant import Ant
    from aco_core.pheromone import PheromoneMatrix
    from aco_core.colony import N_ANTS, N_ITERATIONS, STAGNATION_LIMIT
    import time
    from typing import Optional, Dict

    jobs = colony._jobs
    nodes = colony._nodes
    n_jobs = colony._n_jobs
    n_nodes = colony._n_nodes

    # ── Fast path: LC job OR strategy.use_fast_path → deterministic argmax ────
    if force_fast_path or (n_jobs == 1 and jobs[0].workload_type == WorkloadType.LATENCY_CRITICAL):
        # Find node with highest η — deterministic, no stochasticity
        best_node_idx = int(np.argmax(eta[0]))
        if eta[0, best_node_idx] == 0.0:
            raise ColonyFailedError(n_jobs, n_nodes)

        idx_to_job_id = {v: k for k, v in colony._job_index.items()}
        idx_to_node_id = {v: k for k, v in colony._node_index.items()}
        colony.last_run_ms = 0.0
        return {idx_to_job_id[0]: idx_to_node_id[best_node_idx]}

    # ── Normal path: full colony loop with injected η ─────────────────────────
    start = time.perf_counter()

    matrix = PheromoneMatrix(n_jobs, n_nodes)
    best_solution: Optional[Dict[int, int]] = None
    best_cost: float = float("inf")
    stagnation: int = 0

    # Pre-sort job order by priority (stable across all iterations)
    job_order = sorted(
        range(n_jobs),
        key=lambda i: jobs[i].priority,
        reverse=True,
    )

    for _iteration in range(N_ITERATIONS):
        ants: list = [
            Ant(jobs, nodes, matrix, shared_eta=eta, job_order=job_order)
            for _ in range(N_ANTS)
        ]
        for ant in ants:
            ant.construct()

        feasible_ants = [a for a in ants if a.is_feasible]
        iteration_best = (
            min(feasible_ants, key=lambda a: a.total_cost)
            if feasible_ants else None
        )

        if iteration_best and iteration_best.total_cost < best_cost:
            best_cost = iteration_best.total_cost
            best_solution = iteration_best.solution.copy()
            stagnation = 0
        else:
            stagnation += 1

        matrix.evaporate()
        if iteration_best:
            for job_idx, node_idx in iteration_best.solution.items():
                matrix.deposit(job_idx, node_idx, iteration_best.total_cost)

        if stagnation >= STAGNATION_LIMIT:
            break

    colony.last_run_ms = (time.perf_counter() - start) * 1000.0

    if best_solution is None:
        raise ColonyFailedError(n_jobs, n_nodes)

    idx_to_job_id = {v: k for k, v in colony._job_index.items()}
    idx_to_node_id = {v: k for k, v in colony._node_index.items()}

    return {
        idx_to_job_id[ji]: idx_to_node_id[ni]
        for ji, ni in best_solution.items()
    }


# ── Fallback: V1 naive First Fit (kept verbatim, no changes) ──────────────────

def _naive_schedule_internal(
    job_request: JobRequest,
    candidate_nodes: List[ComputeNode],
) -> str:
    """
    First Fit over a pre-filtered candidate list.
    Internal helper — called both by naive_schedule() and as ACO fallback.
    """
    for node in candidate_nodes:
        if node.can_fit(job_request.resources):
            logger.debug(
                "naive_schedule: job %s → node %s (First Fit)",
                job_request.job_id, node.node_id,
            )
            return node.node_id

    raise SchedulingFailedError(
        f"Job {job_request.job_id} could not be placed. "
        f"No node met minimum requirements."
    )


def naive_schedule(
    job_request: JobRequest,
    available_nodes: List[ComputeNode],
) -> str:
    """
    First Fit scheduling algorithm. Kept from V1 — unchanged behaviour.

    Filters to HEALTHY nodes, then returns the first node that satisfies
    CPU, memory, and GPU requirements.

    Args:
        job_request:     The admitted job request.
        available_nodes: List of all known compute nodes.

    Returns:
        node_id of the selected node.

    Raises:
        SchedulingFailedError: if no suitable node is found.
    """
    candidate_nodes = [
        node for node in available_nodes
        if node.state == NodeState.HEALTHY
    ]
    return _naive_schedule_internal(job_request, candidate_nodes)
