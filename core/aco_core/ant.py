"""
aco_core/ant.py
───────────────
One ant: constructs one complete placement solution for all jobs.

What does an ant do?
─────────────────────
An ant represents one independent exploration of the search space.
It walks through the job list and for each job probabilistically
selects a node — not always the best one, but more likely to choose
better ones. This stochasticity is what gives ACO its power: 20 ants
exploring slightly different solutions, then learning from the best.

The two inputs to every decision
──────────────────────────────────
1. Pheromone trail (τ)  — what did previous ants learn?
   Encoded in the shared PheromoneMatrix. High τ[i][j] means past ants
   found it good to place job i on node j.

2. Heuristic desirability (η) — what does domain knowledge say right now?
   Computed fresh each time an ant is created from the current node state
   (utilisation, cost, headroom). η does NOT change during one ant's walk,
   but it changes across iterations as nodes get more loaded.

The selection formula
──────────────────────
P(place job i on node j) = (τ[i][j]^α × η[i][j]^β) / Σ_k(τ[i][k]^α × η[i][k]^β)

  α = 1.0: pheromone exponent — how much historical learning drives choice.
  β = 2.0: heuristic exponent — how much domain knowledge drives choice.

  β > α means heuristic dominates early (iteration 0, all τ equal).
  As iterations progress and τ differentiates, exploitation increases.

The heuristic η breakdown
──────────────────────────
η[i][j] = resource_headroom × cost_gate × workload_affinity × urgency

  resource_headroom:  How much free capacity does node j have for job i?
                      min(cpu_headroom, mem_headroom), capped at 1.0.
                      Ensures ants prefer nodes with room to breathe.

  cost_gate:          Can job i afford node j?
                      1.0 if within budget (or no ceiling set).
                      Soft taper from 1.0 → 0.0 over a 20% overage band.
                      Hard 0.0 if node cost exceeds ceiling by > 20%.
                      (Previously a hard binary cliff — causes bizarre placement
                       decisions near the budget boundary. Now gracefully penalises
                       near-budget nodes without making them completely invisible.)

  workload_affinity:  Is node j the right hardware type for job i?
                      1.5 if GPU node + BATCH (ideal pairing).
                      0.5 if GPU node + LATENCY_CRITICAL (wasteful pairing).
                      0.0 if architecture mismatch (hard constraint).
                      1.0 for everything else.

  urgency:            How important is job i relative to others?
                      1.0 + (priority / 100). Range: [1.01, 2.0].
                      Base of 1.0 ensures even priority-1 jobs have signal.

Zero η = impossible arc
────────────────────────
If η[i][j] = 0.0, the denominator contribution is 0.0, so
P(job i → node j) = 0.0 exactly. The ant will never choose this arc.
This is the correct way to handle infeasible placements in ACO —
not by excluding nodes from the list (which breaks matrix indexing)
but by making their probability zero through the η term.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from orchestrator.shared.models import (
    ComputeNode,
    JobRequest,
    NodeArch,
    WorkloadType,
)
from aco_core.pheromone import PheromoneMatrix

# ── ACO hyperparameters ────────────────────────────────────────────────────────

ALPHA: float = 1.0
"""Pheromone influence exponent.
τ^ALPHA: how much historical learning (exploitation) drives selection.
ALPHA=1.0 is linear — standard starting point.
"""

BETA: float = 2.0
"""Heuristic influence exponent.
η^BETA: how much domain knowledge (exploration) drives selection.
BETA=2.0 squares each η value, amplifying the gap between good and
mediocre nodes. A node with η=0.9 vs η=0.3 goes from 3× to 9× more
likely to be selected after squaring. This is desirable — the scheduler
has real domain knowledge that should be trusted.
"""

# ── Workload affinity multipliers ──────────────────────────────────────────────

AFFINITY_GPU_BATCH: float = 1.5
"""GPU node + BATCH job: ideal pairing.
ML training jobs are designed for GPU; placing them here maximises
GPU utilisation and matches the hardware investment.
"""

AFFINITY_GPU_LATENCY_CRITICAL: float = 0.5
"""GPU node + LATENCY_CRITICAL job: wasteful pairing.
Inference servers typically use <5% GPU. A GPU node costs 5–10×
a CPU node. Penalising this pairing steers latency-critical jobs
to cheaper, appropriately-sized nodes.
"""

AFFINITY_DEFAULT: float = 1.0
"""All other job-node combinations: neutral."""

# ── Cost constants ─────────────────────────────────────────────────────────────

INFEASIBLE_PENALTY: float = 1e9
"""Added to total_cost when any job goes unplaced.
Q / 1e9 ≈ 0 pheromone deposit — effectively blacklists the solution.
Large enough to always dominate any feasible solution cost.
"""

ETA_EPSILON: float = 1e-9
"""Tiny floor to prevent ZeroDivisionError in headroom ratios.
Used as: node.available_cpu_cores / max(job.resources.cpu_cores_min, ETA_EPSILON)
Keeps the formula safe when a resource value is exactly 0.
"""


class Ant:
    """
    Constructs one complete placement solution using pheromone + heuristic.

    Lifecycle:
        1. __init__()     → pre-compute η matrix from current cluster state.
        2. construct()    → walk jobs in priority order, select a node for each.
        3. Read results:  → ant.solution, ant.total_cost, ant.is_feasible.

    The ant is single-use: create a new Ant instance for each construction.
    Do not call construct() twice on the same ant.

    Attributes:
        solution    : Dict[int, int] — job_idx → node_idx for all placed jobs.
        total_cost  : float          — sum of pro-rated node costs for this solution.
        is_feasible : bool           — True if every job was successfully placed.
    """

    def __init__(
        self,
        jobs: List[JobRequest],
        nodes: List[ComputeNode],
        matrix: PheromoneMatrix,
        shared_eta: Optional[np.ndarray] = None,
        job_order: Optional[List[int]] = None,
    ) -> None:
        """
        Initialise the ant and pre-compute the full heuristic matrix η.

        Why pre-compute η in __init__ (not inside construct)?
            η depends on job requirements and current node state.
            It does NOT change during one ant's walk (the ant does not
            update node state as it places jobs — that would create
            inter-ant dependencies and break the independence assumption).
            Computing it once in __init__ avoids n_jobs redundant computations
            inside the select loop.

        Performance optimisation (used by Colony):
            When the Colony creates 20 ants per iteration, η is identical
            for all ants in the same iteration (same jobs, same node state).
            Colony can pass `shared_eta` — a pre-computed η array — to skip
            _compute_eta() for all but the first ant. This reduces 100 η
            computations to 1 per colony.run() call (~10× speedup).

            Similarly, `job_order` (sorted indices by priority) is stable
            across all iterations and can be pre-computed once by Colony.

        Args:
            jobs:        Ordered list of JobRequest objects.
            nodes:       Ordered list of ComputeNode objects.
            matrix:      Shared PheromoneMatrix (read-only for this ant).
            shared_eta:  Optional pre-computed η array (n_jobs × n_nodes).
                         If provided, skips _compute_eta(). Must not be mutated.
            job_order:   Optional pre-sorted job indices (priority descending).
                         If provided, construct() skips the sort step.
        """
        self._jobs = jobs
        self._nodes = nodes
        self._matrix = matrix
        self._n_jobs = len(jobs)
        self._n_nodes = len(nodes)

        # Use shared η if provided (Colony path), else compute from scratch
        self._eta: np.ndarray = (
            shared_eta if shared_eta is not None else self._compute_eta()
        )

        # Pre-computed job order (Colony path) or None (compute in construct)
        self._job_order: Optional[List[int]] = job_order

        # Results — populated by construct()
        self.solution: Dict[int, int] = {}
        self.total_cost: float = 0.0
        self.is_feasible: bool = False

    # ── Heuristic computation ──────────────────────────────────────────────────

    def _compute_eta(self) -> np.ndarray:
        """
        Build the full heuristic matrix η of shape (n_jobs, n_nodes).

        Each cell η[i][j] answers: "How good is it to place job i on node j,
        ignoring what past ants have done?"

        Algorithm:
            For each (job i, node j) pair:
                If node cannot physically fit the job → η = 0.0 (hard gate).
                Otherwise → η = product of four sub-scores.

        Implementation note:
            Build as a nested Python list, then convert to numpy array once
            at the end. This is cleaner than filling np.zeros() with
            indexed assignments inside a nested loop, and the performance
            difference is negligible for our matrix sizes.

        Returns:
            np.ndarray of dtype float64, shape (n_jobs, n_nodes).
        """
        rows: List[List[float]] = []

        for job in self._jobs:
            row: List[float] = []
            for node in self._nodes:
                # Hard feasibility gate first — cheapest check
                if not node.can_fit(job.resources):
                    row.append(0.0)
                    continue

                # Compute the four sub-scores
                headroom = self._resource_headroom_score(job, node)
                cost     = self._cost_score(job, node)
                affinity = self._workload_affinity_score(job, node)
                urgency  = self._urgency_score(job)

                row.append(headroom * cost * affinity * urgency)

            rows.append(row)

        return np.array(rows, dtype=np.float64)

    @staticmethod
    def _resource_headroom_score(job: JobRequest, node: ComputeNode) -> float:
        """
        How much free capacity does this node have relative to the job's needs?

        Formula:
            cpu_ratio = available_cpu / requested_cpu   (capped at 1.0)
            mem_ratio = available_mem / requested_mem   (capped at 1.0)
            score     = min(cpu_ratio, mem_ratio)

        Why min(cpu, mem)?
            The bottleneck resource determines placement quality.
            A node with 20× CPU headroom but 1.01× memory headroom is
            nearly full — we should reflect that tightness in the score.
            Taking the minimum captures the most constrained dimension.

        Why cap at 1.0?
            We do not reward a node extra for having 100× the needed
            resources. Beyond "fits with headroom," the gain is already
            captured by workload affinity. Capping prevents one very
            empty node from dominating regardless of other factors.

        Example:
            node has 8.0 available CPU, job needs 4.0 → ratio = 2.0 → 1.0
            node has 20.0 available mem, job needs 16.0 → ratio = 1.25 → 1.0
            score = min(1.0, 1.0) = 1.0  (node has plenty of both)

            node has 4.1 available CPU, job needs 4.0 → ratio = 1.025 → 1.0
            node has 16.5 available mem, job needs 16.0 → ratio = 1.03 → 1.0
            score = 1.0  (just barely fits — still full score)

        Returns:
            float in [0.0, 1.0]
        """
        cpu_ratio = node.available_cpu_cores / max(
            job.resources.cpu_cores_min, ETA_EPSILON
        )
        mem_ratio = node.available_memory_gb / max(
            job.resources.memory_gb_min, ETA_EPSILON
        )

        # Guard against negative available resources (over-allocation edge case)
        cpu_ratio = max(cpu_ratio, 0.0)
        mem_ratio = max(mem_ratio, 0.0)

        return min(min(cpu_ratio, mem_ratio), 1.0)

    @staticmethod
    def _cost_score(job: JobRequest, node: ComputeNode) -> float:
        """
        Does this node respect the job's cost ceiling?

        Previous behaviour (hard cliff):
            Within budget → 1.0,  over budget → 0.0 immediately.
            Problem: a node at $0.51 with a $0.50 ceiling scores the same
            as a $50 node — bizarre placement decisions at the boundary.

        New behaviour (soft taper over a 20% overage band):
            No ceiling                     → 1.0  (no constraint)
            cost ≤ ceiling                 → 1.0  (fully within budget)
            ceiling < cost ≤ ceiling×1.20  → linear taper 1.0 → 0.0
                                             (visible but increasingly penalised)
            cost > ceiling×1.20            → 0.0  (hard exclude)

        The 20% taper matches typical cloud spot-price variance. A job with a
        $1.00 ceiling still considers a $1.10 node (score≈0.5) but ignores a
        $1.50 node completely.

        Returns:
            float in [0.0, 1.0]
        """
        if job.cost_ceiling_usd is None:
            return 1.0
        ceiling = job.cost_ceiling_usd
        node_cost = node.cost_profile.cost_per_hour_usd
        if node_cost <= ceiling:
            return 1.0
        # 20% soft taper band above the ceiling
        overage_limit = ceiling * 1.20
        if node_cost > overage_limit:
            return 0.0
        # Linear taper: 1.0 at ceiling, 0.0 at ceiling×1.20
        overage_frac = (node_cost - ceiling) / (overage_limit - ceiling)
        return 1.0 - overage_frac

    @staticmethod
    def _workload_affinity_score(job: JobRequest, node: ComputeNode) -> float:
        """
        Is this node's hardware type a good match for this job's class?

        Architecture check (hard constraint, checked first):
            If job.arch_required is set and node.arch doesn't match → 0.0.
            This is a hard gate: some binaries only run on x86_64 or ARM64.
            No pheromone level can override this physical incompatibility.

        Affinity multipliers:
            GPU_NODE + BATCH             → 1.5  (ideal: GPU for ML training)
            GPU_NODE + LATENCY_CRITICAL  → 0.5  (wasteful: expensive GPU idle)
            GPU_NODE + STREAM            → 1.0  (neutral: some stream jobs use GPU)
            CPU nodes (any workload)     → 1.0  (neutral: general purpose)

        Why 1.5 for GPU+BATCH?
            This is the target affinity — the scheduler actively steers
            BATCH jobs to GPU nodes. A 50% boost makes GPU nodes 1.5×
            more attractive for BATCH vs any other node.

        Why 0.5 for GPU+LATENCY_CRITICAL?
            Inference servers don't use GPU intensively. A GPU node costs
            5–10× a CPU node. The 0.5 penalty halves the effective η,
            steering latency-critical jobs to cheaper CPU nodes.
            The penalty is soft (not 0.0) because: if no CPU nodes are
            available, an ant CAN still place a latency-critical job on
            a GPU node — it just won't prefer it.

        Returns:
            float in {0.0, 0.5, 1.0, 1.5}
        """
        # Architecture hard gate
        if job.arch_required is not None and node.arch != job.arch_required:
            return 0.0

        # GPU node affinity
        if node.arch == NodeArch.GPU_NODE:
            if job.workload_type == WorkloadType.BATCH:
                return AFFINITY_GPU_BATCH
            if job.workload_type == WorkloadType.LATENCY_CRITICAL:
                return AFFINITY_GPU_LATENCY_CRITICAL

        return AFFINITY_DEFAULT

    @staticmethod
    def _urgency_score(job: JobRequest) -> float:
        """
        How important is this job relative to others?

        Formula:
            urgency = 1.0 + (job.priority / 100.0)

        Range: [1.01, 2.0] for priority in [1, 100].

        Why add 1.0 (not just priority / 100)?
            Without the base, priority=1 gives urgency=0.01.
            Multiplied through the η product, this makes low-priority
            jobs nearly invisible to all ants — they'd almost never get
            placed. The base of 1.0 ensures every job has a meaningful
            nonzero urgency. The gradient is gentle: priority-100 is
            exactly twice as urgent as priority-1 (2.0 vs 1.01).

        Note on deadline urgency (future enhancement):
            job.deadline_epoch could boost urgency when deadline is close:
            urgency += (1.0 / max(deadline_epoch - now, 1)) × DEADLINE_WEIGHT
            Not implemented in Phase 2 — added in Phase 5 when the full
            scheduler has a clock reference.

        Returns:
            float in [1.01, 2.0]
        """
        return 1.0 + (job.priority / 100.0)

    # ── Node selection ─────────────────────────────────────────────────────────

    def _select_node(self, job_idx: int) -> Optional[int]:
        """
        Probabilistically select a node for the given job using roulette-wheel.

        This is the core stochastic step. The ant does NOT always pick
        the best node — it samples from a weighted distribution. Higher-η,
        higher-τ nodes are more likely to be chosen, but occasionally
        a suboptimal node is chosen. This exploration is essential: without
        it, all ants would produce the same solution and the colony
        would not learn.

        Formula:
            tau_row    = pheromone_matrix.get_row(job_idx)   # shape (n_nodes,)
            eta_row    = self._eta[job_idx]                  # shape (n_nodes,)
            numerators = (tau_row ** ALPHA) * (eta_row ** BETA)
            total      = numerators.sum()
            if total == 0.0: return None
            probabilities = numerators / total
            chosen = np.random.choice(n_nodes, p=probabilities)

        NumPy operations breakdown:
            tau_row ** ALPHA:   elementwise power (ALPHA=1 → identity, kept for clarity).
            eta_row ** BETA:    elementwise square (BETA=2 → amplifies η differences).
            *:                  elementwise multiply — combines pheromone + heuristic.
            .sum():             scalar sum — denominator for normalisation.
            /:                  elementwise divide → valid probability vector (sums to 1).
            np.random.choice:   roulette-wheel: samples one index weighted by probabilities.

        Critical guard — zero total:
            If ALL nodes have η=0.0 (all infeasible or all over budget),
            then all numerators are 0.0, total=0.0.
            Calling np.random.choice with a zero-sum probability vector
            raises ValueError. The guard catches this before the call.

        Args:
            job_idx: Position of the job in self._jobs (row index in matrix).

        Returns:
            int: chosen node index (column in matrix), or None if no feasible node.
        """
        # Read pheromone row for this job (VIEW — do not mutate)
        tau_row: np.ndarray = self._matrix.get_row(job_idx)
        eta_row: np.ndarray = self._eta[job_idx]

        # Combine pheromone and heuristic (elementwise)
        numerators: np.ndarray = (tau_row ** ALPHA) * (eta_row ** BETA)
        total: float = float(numerators.sum())

        # Guard: all nodes infeasible or all over budget
        if total == 0.0:
            return None

        # Normalise to probability distribution
        probabilities: np.ndarray = numerators / total

        # Roulette-wheel selection via cumulative sum + searchsorted.
        # This is 7× faster than np.random.choice(p=...) for small arrays.
        #
        # How it works:
        #   cumsum = [0.05, 0.35, 0.55, 0.80, 1.00]  (from probabilities)
        #   u      = 0.42  (random float in [0, 1))
        #   searchsorted finds the first index where cumsum[i] >= u → index 2
        #   → node 2 is selected
        #
        # np.searchsorted is O(log n) on sorted data and vectorised in C.
        cumsum: np.ndarray = np.cumsum(probabilities)
        chosen: int = int(np.searchsorted(cumsum, np.random.random()))
        # Clamp to valid range (edge case: floating-point rounding can push
        # cumsum[-1] slightly below 1.0, causing searchsorted to return n_nodes)
        chosen = min(chosen, self._n_nodes - 1)
        return chosen

    # ── Solution construction ──────────────────────────────────────────────────

    def construct(self) -> bool:
        """
        Build a complete solution: select a node for every job.

        Algorithm:
            1. Determine job processing order (highest priority first).
               High-priority jobs get first pick of nodes. Without ordering,
               a low-priority job might claim the last GPU before a
               high-priority job that needs it.

            2. For each job (in priority order):
               a. Call _select_node(job_idx).
               b. If None returned → mark is_feasible=False, continue.
                  We don't abort — processing remaining jobs gives us
                  a partial solution that the colony can still analyse.
               c. Store job_idx → node_idx in self.solution.

            3. Compute total_cost:
               - For each placed (job, node) pair: pro-rated hourly cost
                 = node.cost_per_hour_usd × (job.cpu_min / node.total_cpu)
               - If not fully feasible: add INFEASIBLE_PENALTY.

        Pro-rated cost rationale:
            A job using 4 of 16 CPUs on a $2/hr node "costs"
            4/16 × $2 = $0.50/hr. This reflects the true resource cost
            more accurately than counting the full node rate.
            It also incentivises efficient packing — using more of a node
            doesn't increase the job's cost allocation.

        Returns:
            bool: True if all jobs were placed (is_feasible).

        Post-conditions:
            self.solution populated for all successfully placed jobs.
            self.total_cost set.
            self.is_feasible set.
        """
        # 1. Determine priority order (high → low).
        #    Use pre-computed order if Colony injected one (avoids re-sorting
        #    the same list 100× across N_ANTS × N_ITERATIONS).
        job_order = self._job_order if self._job_order is not None else sorted(
            range(self._n_jobs),
            key=lambda i: self._jobs[i].priority,
            reverse=True,
        )

        all_placed = True

        # 2. Place each job
        for job_idx in job_order:
            node_idx = self._select_node(job_idx)
            if node_idx is None:
                # No feasible node for this job
                all_placed = False
                # Don't add to solution — leave gap
            else:
                self.solution[job_idx] = node_idx

        self.is_feasible = all_placed

        # 3. Compute total cost
        total = 0.0
        for job_idx, node_idx in self.solution.items():
            job  = self._jobs[job_idx]
            node = self._nodes[node_idx]
            cpu_fraction = job.resources.cpu_cores_min / max(
                node.total_cpu_cores, ETA_EPSILON
            )
            total += node.cost_profile.cost_per_hour_usd * cpu_fraction

        if not self.is_feasible:
            total += INFEASIBLE_PENALTY

        self.total_cost = total
        return self.is_feasible

    def __repr__(self) -> str:
        placed = len(self.solution)
        return (
            f"Ant(placed={placed}/{self._n_jobs}, "
            f"cost={self.total_cost:.4f}, "
            f"feasible={self.is_feasible})"
        )
