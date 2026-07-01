"""
aco_core/colony.py
──────────────────
The Colony: orchestrates all ants across all iterations.

How the colony works
─────────────────────
The colony is the outer loop of the ACO algorithm. It:

  1. Creates a shared PheromoneMatrix (the colony's long-term memory).
  2. For each iteration:
       a. Spawns N_ANTS ants — each independently constructs a full
          PlacementPlan using the current pheromone levels + heuristic.
       b. Identifies the best feasible ant (lowest total cost).
       c. Evaporates the matrix (global forgetting — prevents lock-in).
       d. Deposits pheromone for the best ant's solution (positive feedback).
  3. After all iterations: returns the best PlacementPlan found across
     ALL iterations (not just the last one).

Why multiple iterations?
  Iteration 1: Pheromone is uniform — ants explore broadly.
  Iteration 2: Slight differentiation — better arcs have a bit more τ.
  Iteration 3–5: Convergence — ants increasingly exploit the best arcs.
  5 iterations balances exploration/exploitation at our problem scale
  (≤20 jobs, ≤10 nodes) within the <8ms latency budget.

Why multiple ants per iteration?
  One ant = one stochastic sample. Its randomness might happen to pick
  a bad node for one job. 20 ants = 20 independent samples — the colony
  takes the best. This is the ACO equivalent of "try many paths, keep
  the shortest."

Two scheduling paths
─────────────────────
1. Fast path (single LATENCY_CRITICAL job):
     Skip the colony loop entirely. Compute η once, pick argmax (the
     deterministically best node). Target: <1ms.
     Why deterministic? Variance in placement quality is unacceptable
     for jobs with strict P99 SLA targets.

2. Normal path (all other cases):
     Full colony: N_ANTS × N_ITERATIONS. Stochastic, learns from past.
     Target: ≤8ms for the benchmark case (20 ants × 5 iter × 10 nodes).

Index management
─────────────────
The PheromoneMatrix and Ant classes work with integer indices (row, col).
The Colony bridges between string IDs (job_id, node_id) from the models
and integer indices in the matrix.

  _job_index  : Dict[str, int] — job_id  → position in self._jobs
  _node_index : Dict[str, int] — node_id → position in self._nodes

Built in __init__ as dict comprehensions: O(n) to build, O(1) to query.
Reversed at the end of run() to translate the best solution back to IDs.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from orchestrator.shared.models import (
    ComputeNode,
    JobRequest,
    PlacementPlan,
    WorkloadType,
)
from aco_core.ant import (
    Ant,
    ETA_EPSILON,
    AFFINITY_GPU_BATCH,
    AFFINITY_GPU_LATENCY_CRITICAL,
    AFFINITY_DEFAULT,
)
from aco_core.pheromone import PheromoneMatrix

# ── Colony hyperparameters ─────────────────────────────────────────────────────

N_ANTS: int = 20
"""Number of ants per iteration.
20 ants gives good coverage of the search space at our problem scale.
More ants → better exploration but higher cost per iteration.
"""

N_ITERATIONS: int = 5
"""Maximum number of iterations.
Each iteration allows the pheromone matrix to differentiate further.
5 iterations is sufficient for ≤20 jobs / ≤10 nodes.
Increase to 10–20 for larger problems (with corresponding latency trade-off).
"""

STAGNATION_LIMIT: int = 3
"""Early-stop: halt if best cost hasn't improved for this many iterations.
Prevents wasting iterations once the colony has converged.
3 means: if iterations 3, 4, and 5 all produce the same best cost, stop.
"""


class ColonyFailedError(Exception):
    """
    Raised when the colony cannot find any feasible placement.

    When is this raised?
        • No ant across any iteration produced a fully feasible solution
          (every ant had at least one job with zero feasible nodes).
        • This typically means the cluster is out of capacity for the
          submitted workload profile.

    Caller contract:
        The orchestration service (Phase 5) must catch ColonyFailedError
        and fall back to the naive first-fit scheduler from V1.
        This ensures the system never silently drops jobs — graceful
        degradation is built into the error handling chain.

    Attributes:
        n_jobs:  Number of jobs that were being scheduled.
        n_nodes: Number of candidate nodes available.
    """

    def __init__(
        self,
        n_jobs: int,
        n_nodes: int,
        message: str = "",
    ) -> None:
        self.n_jobs = n_jobs
        self.n_nodes = n_nodes
        default_msg = (
            f"Colony failed: no feasible placement found for "
            f"{n_jobs} job(s) on {n_nodes} node(s). "
            f"Cluster may be at capacity."
        )
        super().__init__(message or default_msg)


class Colony:
    """
    Runs the full ACO colony and returns the best PlacementPlan.

    Usage:
        colony = Colony(jobs=job_list, nodes=node_list)
        plan   = colony.run()   # Dict[job_id, node_id]

    After run():
        colony.last_run_ms  → wall-clock time of the last run() call.
                               Used in the performance benchmark test.

    Attributes:
        _jobs:        List[JobRequest]   — stable ordering, defines row indices.
        _nodes:       List[ComputeNode]  — stable ordering, defines col indices.
        _job_index:   Dict[str, int]     — job_id  → int index.
        _node_index:  Dict[str, int]     — node_id → int index.
        last_run_ms:  float              — duration of last run() in milliseconds.
    """

    def __init__(
        self,
        jobs: List[JobRequest],
        nodes: List[ComputeNode],
        initial_tau: Optional["np.ndarray"] = None,
    ) -> None:
        """
        Initialise the colony with the jobs to place and candidate nodes.

        Builds O(1) lookup dictionaries for translating between string IDs
        and integer matrix indices.

        Args:
            jobs:        Non-empty list of JobRequest objects. Order is stable
                         throughout the colony's life — defines row indices.
            nodes:       Non-empty list of ComputeNode objects. Order is stable
                         throughout the colony's life — defines column indices.
            initial_tau: Optional 1-D array of shape (n_nodes,) with per-node
                         pheromone priors accumulated across past scheduling
                         calls. When provided, PheromoneMatrix seeds itself
                         with these values instead of uniform TAU_INITIAL=1.0,
                         enabling the colony to benefit from learned history.

        Raises:
            ValueError: if jobs or nodes is empty.

        Index construction:
            _job_index  = {job.job_id: i for i, job in enumerate(jobs)}
            _node_index = {node.node_id: i for i, node in enumerate(nodes)}

            Dict comprehensions: O(n) to build, O(1) to query per lookup.
            The alternative — searching the list each time — would be O(n)
            per lookup, making the end-of-run translation O(n_jobs²).
        """
        if not jobs:
            raise ValueError("Colony requires at least one job.")
        if not nodes:
            raise ValueError("Colony requires at least one node.")

        self._jobs = jobs
        self._nodes = nodes
        self._n_jobs = len(jobs)
        self._n_nodes = len(nodes)
        self._initial_tau = initial_tau   # forwarded to PheromoneMatrix in run()

        # O(1) lookup: string ID → integer position
        self._job_index:  Dict[str, int] = {
            job.job_id: i for i, job in enumerate(jobs)
        }
        self._node_index: Dict[str, int] = {
            node.node_id: i for i, node in enumerate(nodes)
        }

        # Populated after run()
        self.last_run_ms: float = 0.0

    # ── Fast path ─────────────────────────────────────────────────────────────

    def _fast_path_latency_critical(self, job: JobRequest) -> PlacementPlan:
        """
        Single-job LATENCY_CRITICAL fast path: skip colony, pick best node greedily.

        When invoked:
            run() detects len(self._jobs) == 1 AND
            job.workload_type == WorkloadType.LATENCY_CRITICAL.

        Why skip the colony?
            Running 20 ants × 5 iterations adds 3–5ms overhead.
            For latency-critical jobs with strict P99 SLAs, this overhead
            is unacceptable. The fast path gets a placement in <1ms.

        Why argmax (not roulette-wheel)?
            Latency-critical jobs must go to the BEST node, deterministically.
            Variance in placement quality risks SLA violations.
            Exploration is a luxury we can't afford for these jobs.

        Algorithm:
            Reuses Ant's @staticmethod scoring methods directly — no Ant
            object is instantiated (avoids the η matrix allocation).
            Iterates over nodes, computes scalar η for each, returns the
            node with the highest η.

        Args:
            job: The single LATENCY_CRITICAL job to place.

        Returns:
            PlacementPlan with exactly one entry: {job.job_id: node_id}.

        Raises:
            ColonyFailedError: if no node can accommodate the job
                               (all η values are 0.0).
        """
        best_score: float = -1.0
        best_node_id: Optional[str] = None

        for node in self._nodes:
            if not node.can_fit(job.resources):
                continue

            # Reuse static scoring methods — no Ant instantiation needed
            score = (
                Ant._resource_headroom_score(job, node)
                * Ant._cost_score(job, node)
                * Ant._workload_affinity_score(job, node)
                * Ant._urgency_score(job)
            )

            if score > best_score:
                best_score = score
                best_node_id = node.node_id

        if best_node_id is None:
            raise ColonyFailedError(1, self._n_nodes)

        return {job.job_id: best_node_id}

    # ── Main colony loop ───────────────────────────────────────────────────────

    def run(self) -> PlacementPlan:
        """
        Execute the full ACO colony and return the best PlacementPlan found.

        This is the primary public API of the ACO engine.

        Returns:
            PlacementPlan = Dict[str, str] mapping job_id → node_id.
            Contains an entry for every job in self._jobs.

        Raises:
            ColonyFailedError: if no feasible solution was found in any
                               iteration. Caller should fall back to naive
                               first-fit.

        Algorithm overview:
            See module docstring for the full conceptual explanation.
            Implementation notes inline below.
        """
        # ── Fast path check ─────────────────────────────────────────────────
        if (
            self._n_jobs == 1
            and self._jobs[0].workload_type == WorkloadType.LATENCY_CRITICAL
        ):
            return self._fast_path_latency_critical(self._jobs[0])

        # ── Normal path: full colony loop ───────────────────────────────────
        start = time.perf_counter()

        # Seed the matrix with cross-call pheromone history if provided.
        # This is what gives ACO its long-term learning: each run starts
        # informed by which nodes have performed well in past calls.
        matrix = PheromoneMatrix(self._n_jobs, self._n_nodes, self._initial_tau)

        # Track the globally best solution across all iterations
        best_solution: Optional[Dict[int, int]] = None
        best_cost: float = float("inf")
        stagnation: int = 0

        # Pre-compute η ONCE for all iterations — it depends only on job
        # requirements and node state, neither of which changes mid-colony.
        # This eliminates N_ANTS × N_ITERATIONS - 1 = 99 redundant calls.
        shared_eta = Ant(self._jobs, self._nodes, matrix)._eta

        # Pre-compute job priority order (descending) — stable across iterations.
        job_order = sorted(
            range(self._n_jobs),
            key=lambda i: self._jobs[i].priority,
            reverse=True,
        )

        for _iteration in range(N_ITERATIONS):

            # Spawn ants — inject shared eta and pre-sorted order
            ants: List[Ant] = [
                Ant(self._jobs, self._nodes, matrix,
                    shared_eta=shared_eta, job_order=job_order)
                for _ in range(N_ANTS)
            ]
            for ant in ants:
                ant.construct()

            # Find the best FEASIBLE ant this iteration
            feasible_ants = [a for a in ants if a.is_feasible]
            iteration_best: Optional[Ant] = (
                min(feasible_ants, key=lambda a: a.total_cost)
                if feasible_ants
                else None
            )

            # Update global best
            if iteration_best is not None and iteration_best.total_cost < best_cost:
                best_cost     = iteration_best.total_cost
                best_solution = iteration_best.solution.copy()
                stagnation    = 0
            else:
                stagnation += 1

            # Evaporate BEFORE deposit (decay always precedes reinforcement)
            matrix.evaporate()

            # Deposit pheromone for this iteration's best solution
            if iteration_best is not None:
                for job_idx, node_idx in iteration_best.solution.items():
                    matrix.deposit(job_idx, node_idx, iteration_best.total_cost)

            # Early stopping: no improvement for STAGNATION_LIMIT iterations
            if stagnation >= STAGNATION_LIMIT:
                break

        self.last_run_ms = (time.perf_counter() - start) * 1000.0

        # ── Translate result ─────────────────────────────────────────────────
        if best_solution is None:
            raise ColonyFailedError(self._n_jobs, self._n_nodes)

        # Reverse the index dictionaries to map int → string ID
        idx_to_job_id:  Dict[int, str] = {v: k for k, v in self._job_index.items()}
        idx_to_node_id: Dict[int, str] = {v: k for k, v in self._node_index.items()}

        plan: PlacementPlan = {
            idx_to_job_id[job_idx]: idx_to_node_id[node_idx]
            for job_idx, node_idx in best_solution.items()
        }

        return plan

    def __repr__(self) -> str:
        return (
            f"Colony(jobs={self._n_jobs}, nodes={self._n_nodes}, "
            f"last_run_ms={self.last_run_ms:.2f})"
        )
