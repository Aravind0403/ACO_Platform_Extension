"""
orchestrator/control_plane/cost_engine.py
──────────────────────────────────────────
CostEngine: composite node scoring for placement decisions.

What this is
─────────────
The cost engine translates raw node economics and risk into a single
scalar score — the η (eta) heuristic that the ACO colony uses when
deciding where to place a job.

V1 scheduler had zero cost awareness. If two nodes both had free CPUs,
it picked the first one in the list. This leads to:
  • Latency-critical jobs landing on spot instances (→ SLA breach risk)
  • Batch jobs running on expensive on-demand nodes (→ budget waste)
  • No consideration of predicted load (→ nodes already near saturation)

V2 cost engine fixes all three by scoring each (job, node) pair.

The composite score
────────────────────
  score(job, node) = reliability_factor(job, node)
                   × cost_efficiency_factor(job, node)
                   × sla_headroom_factor(job, node)
                   × prediction_factor(job, node)

All four sub-scores are in (0.0, 1.0] and multiplied together.
A score of 0.0 means "never place this job here" (hard gate).
A score close to 1.0 means "excellent fit across all dimensions."

This score feeds directly into the ACO as η[i][j] — it IS the heuristic.
When the ant computes P(job i → node j), it weights by score(i,j)^β.

Sub-scores explained
─────────────────────
1. reliability_factor:
     ON_DEMAND node:         always 1.0
     SPOT + LATENCY_CRITICAL: penalised if interruption_prob > SPOT_PENALTY_THRESHOLD
       score = (1 − interruption_prob)  → [0, 1], drops toward 0 as risk rises
     SPOT + BATCH/STREAM:    mild penalty only; interruption is tolerable
       score = 1 − 0.3 × interruption_prob  → still usable even at 50% risk

2. cost_efficiency_factor:
     Rewards cheaper nodes without completely ignoring expensive ones.
     norm_cost = node.cost_per_hour / MAX_COST_REFERENCE (1.0 USD/hr)
     score = 1 / (1 + norm_cost)   → asymptotic: never zero, rewards cheap

     Why not just 1/cost?  At cost=0.01 → score=100. At cost=0.1 → score=10.
     That dynamic range is too wide; cheap nodes would dominate completely.
     The (1 + norm_cost) denominator flattens it: 0.01→0.99, 0.5→0.67, 1.0→0.5.

3. sla_headroom_factor:
     Does the node have enough spare capacity to meet the job's latency SLA?
     Uses cpu_utilisation_pct (real telemetry if available, allocation otherwise).
     headroom = (100 - cpu_util) / 100   → 0.0 (fully loaded) to 1.0 (empty)
     For LATENCY_CRITICAL: score = headroom (strict — needs real breathing room)
     For BATCH / STREAM:   score = max(headroom, 0.1)  (forgiving — throughput jobs)

4. prediction_factor:
     If a PredictionResult is available for this node, use spike_probability
     to penalise nodes expected to become overloaded during the job's run.
     score = 1 − (spike_probability × SPIKE_PENALTY_WEIGHT)
     SPIKE_PENALTY_WEIGHT = 0.5 → a 100% spike prediction halves the score.
     If no prediction available: score = 1.0 (neutral — no penalty).

Integration
───────────
  Phase 2 (current):
    Colony.run() → passes the existing Ant._workload_affinity_score and
    Ant._resource_headroom_score + Ant._cost_score as sub-components of η.

  Phase 5 (upcoming):
    aco_schedule() in scheduler.py will call CostEngine.score_node() to
    produce a richer η that incorporates prediction and SLA headroom.
    The CostEngine replaces the current per-function approach in ant.py
    with a single authoritative score function.

Standalone use:
    from orchestrator.control_plane.cost_engine import CostEngine
    engine = CostEngine()
    score = engine.score_node(job=job, node=node, prediction=result)
    # Returns float in [0.0, 1.0]
"""

from __future__ import annotations

from typing import Optional

from orchestrator.shared.models import (
    ComputeNode,
    InstanceType,
    JobRequest,
    PredictionResult,
    WorkloadType,
)

# ── Cost engine constants ──────────────────────────────────────────────────────
# Module-level so tests can import and assert against them directly.

MAX_COST_REFERENCE: float = 1.0
"""Normalisation reference for cost efficiency scoring (USD/hour).

Nodes cheaper than this score > 0.5; nodes at this price score 0.5.
$1.00/hr covers typical on-demand prices for medium instances.
Adjust for GPU clusters (p4d.24xlarge = $32/hr → scores 0.03, penalising
expensive GPU nodes for non-GPU jobs as expected).
"""

SPOT_PENALTY_THRESHOLD: float = 0.3
"""Interruption probability above which LATENCY_CRITICAL jobs are hard-gated.

Above this threshold a SPOT node receives score = 0.0 for LC jobs.
0.3 = "30% chance of eviction in the next hour" — unacceptable for SLA-bound
workloads. On-demand nodes are always 0.0 interruption_prob, so they're safe.

For BATCH and STREAM workloads, we apply a soft penalty instead of a hard gate.
"""

SPIKE_PENALTY_WEIGHT: float = 0.5
"""How much a predicted CPU spike reduces the placement score.

score = 1 − (spike_probability × SPIKE_PENALTY_WEIGHT)

At weight=0.5:
  spike_prob = 0.0 → prediction_factor = 1.0  (no penalty)
  spike_prob = 0.5 → prediction_factor = 0.75 (mild penalty)
  spike_prob = 1.0 → prediction_factor = 0.5  (heavy penalty — halves score)

This is intentionally soft: the predictor has uncertainty, and we should
not hard-block placements based on predictions alone. The colony sees
lower η for risky nodes and will prefer alternatives — but if all nodes
are predicted to spike, the colony still picks the least-bad one.
"""

SLA_STRICT_THRESHOLD: float = 0.2
"""Minimum headroom fraction required to pass strict SLA check.

For LATENCY_CRITICAL jobs: if (1 - cpu_util/100) < this threshold,
the node is considered too loaded and receives score = 0.0.
0.2 means: nodes at > 80% CPU utilisation are hard-gated for LC jobs.
"""


class CostEngine:
    """
    Composite node scorer for placement decisions.

    Stateless: all scoring is deterministic given the inputs.
    One engine instance can be shared across all scheduling calls.

    Usage:
        engine = CostEngine()
        score = engine.score_node(job, node)                 # no prediction
        score = engine.score_node(job, node, prediction)     # with predictor output

    Returns:
        float in [0.0, 1.0].
        0.0 = hard gate (never place here).
        1.0 = ideal placement across all dimensions.
    """

    # ── Main scoring entrypoint ────────────────────────────────────────────────

    def score_node(
        self,
        job: JobRequest,
        node: ComputeNode,
        prediction: Optional[PredictionResult] = None,
        *,
        sla_threshold: Optional[float] = None,
        spike_weight: Optional[float] = None,
        spot_threshold: Optional[float] = None,
    ) -> float:
        """
        Compute the composite placement score for (job, node).

        This is the η heuristic fed into the ACO selection formula:
            P(job i → node j) ∝ τ[i,j]^α × score(i,j)^β

        Algorithm:
            1. Compute four independent sub-scores.
            2. Multiply them together.
            3. Return the product (product rule: any zero makes total zero).

        Args:
            job:        The job to be placed.
            node:       The candidate node to score.
            prediction: Optional CPU spike forecast from WorkloadPredictor.
                        If None, prediction_factor defaults to 1.0 (neutral).
            sla_threshold:  Override for SLA_STRICT_THRESHOLD (keyword-only).
                            Used by WorkloadIntentRouter strategies that require
                            tighter or looser headroom gates.
            spike_weight:   Override for SPIKE_PENALTY_WEIGHT (keyword-only).
            spot_threshold: Override for SPOT_PENALTY_THRESHOLD (keyword-only).

        Returns:
            float in [0.0, 1.0].

        Why multiply sub-scores?
            Because the sub-scores are independent checks. Failing ANY one of
            them should drag the total score down proportionally. If the cost
            gate returns 0.0 (node over budget), the whole score is 0.0 — the
            node is effectively invisible to the ACO colony, just like a zero-η
            arc in the original ant.py implementation.
        """
        r = self.reliability_factor(job, node, spot_threshold=spot_threshold)
        if r == 0.0:
            return 0.0   # fast exit — no need to compute the rest

        c = self.cost_efficiency_factor(node)
        s = self.sla_headroom_factor(job, node, sla_threshold=sla_threshold)
        if s == 0.0:
            return 0.0   # fast exit

        p = self.prediction_factor(prediction, spike_weight=spike_weight)

        return r * c * s * p

    # ── Sub-score methods (public for direct testing) ──────────────────────────

    def reliability_factor(
        self,
        job: JobRequest,
        node: ComputeNode,
        *,
        spot_threshold: Optional[float] = None,
    ) -> float:
        """
        Score the node's reliability relative to the job's tolerance for interruption.

        ON_DEMAND nodes always score 1.0 — they cannot be preempted.

        SPOT nodes:
          LATENCY_CRITICAL job:
            → interruption_prob > spot_threshold (default 0.3) → hard gate (0.0)
            → otherwise: score = (1 − interruption_prob)
               e.g. 10% risk → 0.9, 25% risk → 0.75
          BATCH / STREAM job:
            → soft penalty: score = 1 − 0.3 × interruption_prob
               e.g. 50% risk → 0.85 (still usable — batch can be re-queued)
               Rationale: BATCH jobs tolerate interruption because they checkpoint.

        Args:
            job:           The job being placed (determines interruption tolerance).
            node:          The candidate node (cost_profile.instance_type + interruption_prob).
            spot_threshold: Override for SPOT_PENALTY_THRESHOLD (keyword-only).

        Returns:
            float in [0.0, 1.0].
        """
        threshold = SPOT_PENALTY_THRESHOLD if spot_threshold is None else spot_threshold
        cost_profile = node.cost_profile

        if cost_profile.instance_type == InstanceType.ON_DEMAND:
            return 1.0   # on-demand: no interruption risk

        # SPOT node
        prob = cost_profile.interruption_prob

        if job.workload_type == WorkloadType.LATENCY_CRITICAL:
            # Hard gate for high interruption risk on LC jobs
            if prob > threshold:
                return 0.0
            # Soft penalty below threshold
            return max(0.0, 1.0 - prob)

        else:
            # BATCH and STREAM tolerate spot interruption
            # Soft penalty: 30% of the interruption prob is subtracted
            return max(0.0, 1.0 - 0.3 * prob)

    def cost_efficiency_factor(self, node: ComputeNode) -> float:
        """
        Score the node's cost efficiency.

        Formula: 1 / (1 + normalised_cost)
            normalised_cost = cost_per_hour_usd / MAX_COST_REFERENCE

        Properties of this formula:
          • Always in (0.0, 1.0] — never zero, never above 1.0
          • Free node ($0/hr) → 1.0 (maximum reward)
          • $0.10/hr → 1/(1+0.1) ≈ 0.91
          • $0.50/hr → 1/(1+0.5) ≈ 0.67
          • $1.00/hr → 1/(1+1.0) = 0.50
          • $5.00/hr → 1/(1+5.0) ≈ 0.17  (GPU nodes get downweighted)

        Why asymptotic (never zero)?
            Even a very expensive node should not be completely ruled out by cost
            alone — the job might have no other feasible option. The hard budget
            gate (cost_ceiling_usd) in ant.py handles absolute budget exclusions.
            Here we just reward cheaper nodes relatively.

        Args:
            node: The candidate node.

        Returns:
            float in (0.0, 1.0].
        """
        norm_cost = node.cost_profile.cost_per_hour_usd / MAX_COST_REFERENCE
        return 1.0 / (1.0 + norm_cost)

    def sla_headroom_factor(
        self,
        job: JobRequest,
        node: ComputeNode,
        *,
        sla_threshold: Optional[float] = None,
    ) -> float:
        """
        Score whether the node has enough spare capacity to meet the job's SLA.

        Uses node.cpu_utilisation_pct which:
          - Returns real telemetry (actual OS measurement) if available
          - Falls back to allocation-based estimate if no telemetry yet

        For LATENCY_CRITICAL jobs (strict):
          headroom = (100 - cpu_util) / 100.0  → fraction [0.0, 1.0]
          If headroom < sla_threshold (default 0.2): hard gate → return 0.0
          Otherwise: return headroom directly (more headroom = higher score)

        For BATCH and STREAM (forgiving):
          headroom = max((100 - cpu_util) / 100.0, 0.1)
          Floor at 0.1: even a 95%-loaded node gets score 0.1 for batch jobs.
          Batch jobs trade some speed for placement — they can run on busier nodes.

        Why use headroom rather than raw util?
            We want to reward nodes with room to breathe, not just penalise
            busy ones. A node at 50% util has 50% headroom — good.
            A node at 95% util has 5% headroom — risky for LC, acceptable for batch.

        Args:
            job:          The job being placed (determines strictness of SLA check).
            node:         The candidate node (provides cpu_utilisation_pct).
            sla_threshold: Override for SLA_STRICT_THRESHOLD (keyword-only).

        Returns:
            float in [0.0, 1.0].
        """
        threshold = SLA_STRICT_THRESHOLD if sla_threshold is None else sla_threshold
        cpu_util = node.cpu_utilisation_pct       # 0–100, real or estimated
        headroom = (100.0 - cpu_util) / 100.0     # 0.0 to 1.0

        if job.workload_type == WorkloadType.LATENCY_CRITICAL:
            # Strict: hard gate if headroom below threshold
            if headroom < threshold:
                return 0.0
            return headroom

        else:
            # Forgiving: floor at 0.1 so batch jobs always have a nonzero score
            return max(headroom, 0.1)

    @staticmethod
    def prediction_factor(
        prediction: Optional[PredictionResult],
        *,
        spike_weight: Optional[float] = None,
    ) -> float:
        """
        Penalise nodes predicted to spike within the forecast horizon.

        If no prediction is available: return 1.0 (neutral — no penalty).

        Formula:
            score = 1 − (spike_probability × spike_weight × confidence)

        The confidence term dampens the penalty when the predictor is uncertain:
          • confidence=0.1 (cold-start): penalty is 10× smaller → nearly neutral
          • confidence=1.0 (fully trained): full penalty applied
          • spike_probability=0.0: score = 1.0 regardless of confidence

        Example (default spike_weight=0.5):
          spike_prob=0.8, confidence=1.0  → 1 − (0.8×0.5×1.0) = 0.6
          spike_prob=0.8, confidence=0.1  → 1 − (0.8×0.5×0.1) = 0.96 (barely penalised)
          spike_prob=0.0                  → 1.0 always

        Args:
            prediction:  PredictionResult from WorkloadPredictor, or None.
            spike_weight: Override for SPIKE_PENALTY_WEIGHT (keyword-only).

        Returns:
            float in [0.5, 1.0] when prediction is provided (minimum 0.5 because
            even certainty of a spike doesn't warrant a hard block — the predictor
            has error). 1.0 when prediction is None.
        """
        if prediction is None:
            return 1.0

        weight = SPIKE_PENALTY_WEIGHT if spike_weight is None else spike_weight
        penalty = prediction.spike_probability * weight * prediction.confidence
        return max(0.5, 1.0 - penalty)

    # ── Detailed breakdown (for observability / debugging) ─────────────────────

    def score_breakdown(
        self,
        job: JobRequest,
        node: ComputeNode,
        prediction: Optional[PredictionResult] = None,
        *,
        sla_threshold: Optional[float] = None,
        spike_weight: Optional[float] = None,
        spot_threshold: Optional[float] = None,
    ) -> dict:
        """
        Return a dict with each sub-score and the final composite score.

        Used by:
          - GET /metrics endpoint (Phase 9) for operator visibility
          - Unit tests that need to assert per-sub-score values

        Returns:
            {
                "reliability":      float,
                "cost_efficiency":  float,
                "sla_headroom":     float,
                "prediction":       float,
                "composite":        float,
            }
        """
        r = self.reliability_factor(job, node, spot_threshold=spot_threshold)
        c = self.cost_efficiency_factor(node)
        s = self.sla_headroom_factor(job, node, sla_threshold=sla_threshold)
        p = self.prediction_factor(prediction, spike_weight=spike_weight)

        return {
            "reliability": r,
            "cost_efficiency": c,
            "sla_headroom": s,
            "prediction": p,
            "composite": r * c * s * p,
        }

    def __repr__(self) -> str:
        return (
            f"CostEngine("
            f"spot_threshold={SPOT_PENALTY_THRESHOLD}, "
            f"spike_weight={SPIKE_PENALTY_WEIGHT}, "
            f"sla_threshold={SLA_STRICT_THRESHOLD})"
        )
