"""
aco_core/pheromone.py
─────────────────────
The pheromone matrix: the ACO colony's shared, persistent memory.

What is pheromone?
──────────────────
In nature, ants deposit chemical pheromone on paths they walk.
Shorter paths get reinforced more (ants traverse them faster, deposit more).
Over time the colony converges on the optimal path — without any ant
having a global view of the network.

In this scheduler:
  • "Path"   = assigning a specific job to a specific node.
  • "Better" = lower total placement cost (cheaper node, more headroom).
  • τ[i][j]  = pheromone on the arc "place job i on node j".

Two forces balance each other:
  1. Evaporation  — global forgetting. All pheromone decays each iteration.
                    Prevents the colony from locking onto a suboptimal early
                    solution and never trying anything else.
  2. Deposit      — positive reinforcement. After each iteration, the best
                    solution gets extra pheromone. Better solutions get more.
                    Over iterations, good arcs accumulate high τ values,
                    making them increasingly attractive to all ants.

Matrix layout
─────────────
  Shape : (n_jobs, n_nodes)
  τ[i][j]: pheromone level for placing the i-th job on the j-th node.
  Indices are positional integers maintained by the Colony class.
  The matrix itself knows nothing about job IDs or node IDs —
  that translation is the Colony's responsibility.

NumPy design choices
────────────────────
  • float64 throughout — float32 accumulates measurable rounding drift
    over 5+ iterations of multiply-then-add cycles.
  • In-place operations (*=, +=) — avoid allocating new arrays on the
    hot path. For a 20×10 matrix this saves ~200 float64 allocations
    per iteration.
  • np.clip(..., out=self._matrix) — clips in-place, same reason.
  • .copy() only in snapshot() — the one place we need a safe copy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

# ── Pheromone constants ────────────────────────────────────────────────────────
# These are module-level so tests can import and assert against them directly.

TAU_INITIAL: float = 1.0
"""Starting pheromone on every arc.
All arcs equal at iteration 0 → ants start with no bias.
"""

TAU_MIN: float = 0.01
"""Minimum allowed pheromone (the floor).

Without a floor, evaporation drives some arcs to ≈0, making them
permanently invisible to ants even if they are the only feasible option.
TAU_MIN keeps every arc "alive" — any arc can be rediscovered if the
colony's current solution degrades.

Analogy: even rarely-used hiking trails don't vanish completely.
"""

TAU_MAX: float = 10.0
"""Maximum allowed pheromone (the ceiling).

Without a ceiling, one very good early solution accumulates pheromone
unboundedly, crowding out exploration. TAU_MAX ensures diversity is
maintained even after the colony has converged.
"""

EVAPORATION_RATE: float = 0.1
"""ρ (rho): fraction of pheromone that evaporates each iteration.

τ_new = τ_old × (1 − ρ)

ρ = 0.1 means 10% decay per iteration.
After 5 iterations with no deposit, a cell starting at 1.0 becomes:
  1.0 × 0.9^5 ≈ 0.59 (still above TAU_MIN=0.01 — plenty of signal left).

Higher ρ → faster forgetting → more exploration (risks oscillation).
Lower ρ  → slower forgetting → more exploitation (risks premature convergence).
0.1 is a well-established default for scheduling ACO problems.
"""

Q: float = 1.0
"""Deposit numerator.

Deposit amount for one arc = Q / solution_cost.
A solution with cost=0.5 deposits 2.0; cost=2.0 deposits 0.5.
Cheaper (better) solutions reinforce their arcs more strongly.

Q=1.0 is a neutral scaling factor. Increasing Q amplifies learning speed
(ants converge faster but risk overshooting). Decreasing Q slows learning.
"""


class PheromoneMatrix:
    """
    A 2D numpy array τ[n_jobs][n_nodes] storing pheromone levels.

    Used by:
        Ant._select_node()  → reads get_row() to build selection probabilities.
        Colony.run()        → calls evaporate() and deposit() each iteration.
        Tests               → calls snapshot() to inspect internal state.

    Thread safety:
        Not thread-safe. The colony runs ants sequentially in Phase 2.
        If Phase V adds parallel ant threads, add a lock around deposit().
    """

    def __init__(
        self,
        n_jobs: int,
        n_nodes: int,
        initial_tau: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Initialise a pheromone matrix, optionally seeded with prior knowledge.

        Args:
            n_jobs:      Number of jobs (rows). Must be ≥ 1.
            n_nodes:     Number of candidate nodes (columns). Must be ≥ 1.
            initial_tau: Optional 1-D array of shape (n_nodes,) containing the
                         per-node pheromone prior. When provided, every row of
                         the matrix is initialised to these values instead of
                         the uniform TAU_INITIAL=1.0.

                         This enables cross-call learning: OrchestratorService
                         maintains a ``_node_pheromone`` vector that accumulates
                         placement history across scheduling calls. Passing it
                         here seeds the colony with that learned preference so
                         the colony does not start blind on every call.

        Raises:
            ValueError: if either dimension < 1, or if initial_tau has wrong shape.
        """
        if n_jobs < 1 or n_nodes < 1:
            raise ValueError(
                f"PheromoneMatrix requires n_jobs≥1 and n_nodes≥1, "
                f"got n_jobs={n_jobs}, n_nodes={n_nodes}"
            )
        self._n_jobs = n_jobs
        self._n_nodes = n_nodes

        if initial_tau is not None:
            if initial_tau.shape != (n_nodes,):
                raise ValueError(
                    f"initial_tau must have shape ({n_nodes},), got {initial_tau.shape}"
                )
            # Broadcast: all job rows start with the same per-node prior
            row = np.clip(initial_tau.astype(np.float64), TAU_MIN, TAU_MAX)
            self._matrix: NDArray[np.float64] = np.tile(row, (n_jobs, 1))
        else:
            self._matrix: NDArray[np.float64] = np.full(
                (n_jobs, n_nodes), TAU_INITIAL, dtype=np.float64
            )

    # ── Core operations ────────────────────────────────────────────────────────

    def evaporate(self) -> None:
        """
        Apply pheromone evaporation to the entire matrix in-place.

        Formula applied to every cell:
            τ[i][j] = clip( τ[i][j] × (1 − ρ),  TAU_MIN,  TAU_MAX )

        Two steps, both in-place:
            Step 1:  self._matrix *= (1.0 - EVAPORATION_RATE)
                     Multiplies all 200 values (20×10) in one vectorised op.
                     No new array is allocated.

            Step 2:  np.clip(self._matrix, TAU_MIN, TAU_MAX, out=self._matrix)
                     The `out=self._matrix` argument writes the clipped values
                     back into the same memory buffer — again, no allocation.

        Why evaporate BEFORE deposit (not after)?
            Evaporate first → then deposit the best solution on top.
            Result: the best solution always "fights against" decay.
            If we deposited first then evaporated, the freshly deposited
            pheromone would be immediately reduced — a needless weakening
            of the learning signal.

        Assertion (dev mode):
            After evaporation, no cell should be NaN. If NaN appears,
            a previous deposit introduced a corrupted value.
        """
        # Step 1: global decay
        self._matrix *= (1.0 - EVAPORATION_RATE)

        # Step 2: enforce bounds (in-place, no allocation)
        np.clip(self._matrix, TAU_MIN, TAU_MAX, out=self._matrix)

    def deposit(self, job_idx: int, node_idx: int, solution_cost: float) -> None:
        """
        Reinforce the pheromone on one arc (job_idx → node_idx).

        Called by the colony once per arc in the best solution found this
        iteration. Each call reinforces a single cell.

        Formula:
            τ[job_idx][node_idx] += Q / solution_cost

        Args:
            job_idx:       Row (job) index. Must be in [0, n_jobs).
            node_idx:      Column (node) index. Must be in [0, n_nodes).
            solution_cost: Total cost of the full solution this arc belongs to.
                           NOT the cost of this single arc — using the full
                           solution cost means the reinforcement reflects
                           overall placement quality, not one job's contribution.

        Guards:
            • solution_cost ≤ 0 → skip. Avoids division by zero or negative
              deposit (which would corrupt the matrix). A zero-cost solution
              would mean free compute — impossible in practice.
            • After deposit: clip the single cell to TAU_MAX to prevent
              unbounded accumulation on one arc.

        NumPy operation:
            self._matrix[job_idx, node_idx] += Q / solution_cost
            Scalar indexing + scalar add: O(1), no vectorisation needed here
            since we call this once per arc in the best solution.
        """
        if solution_cost <= 0.0:
            return  # Guard: no deposit for zero or negative cost

        deposit_amount = Q / solution_cost
        self._matrix[job_idx, node_idx] += deposit_amount

        # Enforce ceiling on this single cell
        if self._matrix[job_idx, node_idx] > TAU_MAX:
            self._matrix[job_idx, node_idx] = TAU_MAX

    def get_row(self, job_idx: int) -> NDArray[np.float64]:
        """
        Return the pheromone row for one job across all nodes.

        This is read by Ant._select_node() to compute selection probabilities.

        Args:
            job_idx: Row index (job position in the colony's job list).

        Returns:
            1D numpy array of shape (n_nodes,).

        ⚠️ WARNING — this is a VIEW, not a copy:
            numpy basic indexing (self._matrix[i]) returns a view into
            the underlying buffer. The ant MUST NOT modify this array —
            doing so would corrupt the shared pheromone matrix.
            The ant's probability calculation creates its own new arrays
            (tau ** ALPHA, eta ** BETA, etc.) so this is safe by design.

        Why a view and not a copy?
            For a 10-node row, copying 10 float64 values is trivial.
            But the habit matters: returning views avoids unnecessary
            allocations on the hot path. At 100 ants × 20 jobs per run,
            that's 2,000 potential copies saved per colony.run() call.
        """
        return self._matrix[job_idx]

    # ── Inspection & testing ───────────────────────────────────────────────────

    def snapshot(self) -> NDArray[np.float64]:
        """
        Return a deep copy of the current matrix state.

        Use this in tests to:
        • Verify evaporation changed the matrix correctly.
        • Verify deposit increased the right cell.
        • Compare before/after states without the snapshot itself
          being affected by subsequent operations.

        Returns:
            NDArray[np.float64] of shape (n_jobs, n_nodes).
            Mutations to the returned array do NOT affect self._matrix.

        NumPy operation:
            self._matrix.copy()
            .copy() creates a new array with its own buffer.
            This is the one place where allocation is intentional.
        """
        return self._matrix.copy()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, int]:
        """
        The (n_jobs, n_nodes) dimensions of the matrix.

        Used in assertions: assert matrix.shape == (3, 5)
        """
        return (self._n_jobs, self._n_nodes)

    @property
    def n_jobs(self) -> int:
        """Number of job rows."""
        return self._n_jobs

    @property
    def n_nodes(self) -> int:
        """Number of node columns."""
        return self._n_nodes

    def __repr__(self) -> str:
        return (
            f"PheromoneMatrix(n_jobs={self._n_jobs}, n_nodes={self._n_nodes}, "
            f"min={self._matrix.min():.4f}, max={self._matrix.max():.4f}, "
            f"mean={self._matrix.mean():.4f})"
        )
