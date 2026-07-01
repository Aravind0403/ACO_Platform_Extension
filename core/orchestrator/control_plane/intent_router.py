"""
orchestrator/control_plane/intent_router.py
────────────────────────────────────────────
WorkloadIntentRouter: the meta-scheduler layer above the ACO colony.

What this is
─────────────
This is Phase 6 — the component that implements Intent-Based Orchestration.

Instead of treating every job identically and relying solely on CostEngine
scores to differentiate them, the IntentRouter reads the job's declared
intent (workload_type + gpu_required + preemptible + deadline) and selects
a named SchedulingStrategy that configures how the ACO scheduler behaves
for that specific job.

Why this matters
─────────────────
The Ray/GKE/TPU architecture document identified five distinct ML workload
roles, each with completely different hardware and reliability requirements:

  Inference Server → GPU, ON_DEMAND, strict SLA, avoid training nodes
  CPU API Server   → x86/ARM, ON_DEMAND, strict SLA
  Learner          → GPU, spot OK, throughput-focused, stateful
  MCTS Actors      → CPU, spot preferred, ephemeral, high interruption tolerance
  Replay Buffer    → CPU, ON_DEMAND, stateful, can't be interrupted

Our WorkloadType enum (LATENCY_CRITICAL, BATCH, STREAM) maps to these roles,
but the mapping is many-to-one — LATENCY_CRITICAL covers both inference servers
and API servers, which need different placement logic.

The IntentRouter reads the full intent signal:
  workload_type + arch_required + gpu_required + preemptible + deadline_epoch
and selects a SchedulingStrategy that tells aco_schedule() exactly how to behave.

This is the same pattern as:
  - EKS Auto Mode: reads workload intent → provisions right instance type
  - GKE Autopilot: reads workload class → applies right scoring profile
  - Karmada: reads intent → routes to right cluster/scheduler

We implement it at the scheduling-strategy level (not provisioning) since
our cluster is fixed-capacity, but the architectural pattern is identical.

SchedulingStrategy fields
──────────────────────────
  name                  — human-readable identifier (e.g. "GPU_INFERENCE")

  required_arch         — List[NodeArch] or None
                          Hard constraint: only consider nodes with this arch.
                          None = no arch constraint.

  required_instance     — List[InstanceType] or None
                          Hard constraint: only consider these billing models.
                          None = any instance type allowed.

  use_fast_path         — bool
                          True  = deterministic argmax η (skip colony)
                          False = full stochastic colony run

  allow_spot            — bool
                          Convenience flag. If False, SPOT nodes are
                          excluded before scoring (reliability = 0.0 override).
                          Redundant with required_instance=[ON_DEMAND] but
                          explicit for readability.

  sla_strict_threshold  — float (CostEngine override)
                          Minimum headroom fraction required for LATENCY_CRITICAL
                          strategy. Passed to score_node() as sla_threshold kwarg.
                          Default in CostEngine: 0.20

  spike_penalty_weight  — float (CostEngine override)
                          How aggressively spike predictions penalise placement.
                          Default in CostEngine: 0.50

  spot_penalty_threshold — float (CostEngine override)
                          Interruption probability above which SPOT nodes are
                          hard-gated for LC jobs. Default in CostEngine: 0.30

  avoid_workload_types  — List[WorkloadType]
                          Colocation policy: nodes currently running any of
                          these workload types get η = 0.0 for this job.
                          Example: GPU_INFERENCE avoids nodes running BATCH
                          to prevent GPU memory contention.

Routing rules
──────────────
Rules are evaluated top-to-bottom; first match wins. The deadline-urgent
override is applied on top of the matched rule (it modifies the strategy
in place, not returns a new one).

Rule 1: GPU_INFERENCE     LC + gpu_required=True
Rule 2: CPU_SERVING       LC + gpu_required=False
Rule 3: GPU_TRAINING      BATCH + gpu_required=True + not preemptible
Rule 4: PREEMPTIBLE_ACTOR (BATCH or STREAM) + preemptible=True
Rule 5: STATEFUL_STREAM   STREAM + not preemptible
Override: deadline < 60s  → tighten any strategy

Default fallback (no rule matches): GENERIC strategy with all defaults.

Integration
────────────
  Called by: OrchestratorService.submit_job()
             router = WorkloadIntentRouter()
             strategy = router.classify(job_request)
             aco_schedule(..., strategy=strategy)

  Consumes:  JobRequest fields (workload_type, resources.gpu_required,
             preemptible, deadline_epoch, arch_required)

  Produces:  SchedulingStrategy (consumed by aco_schedule + CostEngine)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from orchestrator.shared.models import (
    InstanceType,
    JobRequest,
    NodeArch,
    WorkloadType,
)

# ── How close to the deadline (seconds) before the urgent override kicks in ──
DEADLINE_URGENT_WINDOW_S: float = 60.0
"""
If time_to_deadline < this threshold, the deadline-urgent override applies:
  - Force fast-path (deterministic argmax)
  - Tighten SLA headroom by SLA_DEADLINE_BOOST
  - Avoid all noisy neighbours (BATCH + STREAM)

60 seconds gives enough runway for the scheduler to act before the window closes.
"""

SLA_DEADLINE_BOOST: float = 0.10
"""
Additional headroom fraction added to sla_strict_threshold when deadline is imminent.
0.10 means: if GPU_INFERENCE normally requires 30% headroom, it now requires 40%.
"""


@dataclass
class SchedulingStrategy:
    """
    A complete specification of how the scheduler should behave for one job.

    Produced by WorkloadIntentRouter.classify() and consumed by aco_schedule().
    All fields have sensible defaults — the router overrides only what matters
    for each specific workload class.

    See module docstring for field descriptions.
    """

    name: str

    # ── Hard node constraints (pre-filter before scoring) ─────────────────────
    required_arch: Optional[List[NodeArch]] = None
    required_instance: Optional[List[InstanceType]] = None

    # ── Scheduler path ────────────────────────────────────────────────────────
    use_fast_path: bool = False
    allow_spot: bool = True

    # ── CostEngine threshold overrides ────────────────────────────────────────
    sla_strict_threshold: float = 0.20     # matches CostEngine.SLA_STRICT_THRESHOLD
    spike_penalty_weight: float = 0.50     # matches CostEngine.SPIKE_PENALTY_WEIGHT
    spot_penalty_threshold: float = 0.30   # matches CostEngine.SPOT_PENALTY_THRESHOLD

    # ── Colocation policy ─────────────────────────────────────────────────────
    avoid_workload_types: List[WorkloadType] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"SchedulingStrategy({self.name!r}, "
            f"arch={[a.value for a in self.required_arch] if self.required_arch else 'any'}, "
            f"instance={[i.value for i in self.required_instance] if self.required_instance else 'any'}, "
            f"fast_path={self.use_fast_path}, "
            f"allow_spot={self.allow_spot}, "
            f"sla_threshold={self.sla_strict_threshold})"
        )


class WorkloadIntentRouter:
    """
    Meta-scheduler: reads job intent and selects a SchedulingStrategy.

    Stateless — one instance can be shared across all scheduling calls.
    No external dependencies — pure Python decision tree.

    Usage:
        router = WorkloadIntentRouter()
        strategy = router.classify(job_request)
        # → SchedulingStrategy with all thresholds and constraints set

    The strategy is then passed to aco_schedule() which uses it to:
      1. Pre-filter nodes by required_arch and required_instance
      2. Override CostEngine scoring thresholds
      3. Choose fast-path vs full colony
      4. Apply colocation exclusions
    """

    def classify(self, job: JobRequest) -> SchedulingStrategy:
        """
        Classify a job and return the appropriate SchedulingStrategy.

        Decision tree evaluated top-to-bottom; first match wins.
        Deadline-urgent override applied on top of the matched strategy.

        Args:
            job: The admitted JobRequest to classify.

        Returns:
            SchedulingStrategy configured for this job's workload class.
        """
        strategy = self._match_base_strategy(job)
        self._apply_deadline_override(job, strategy)
        return strategy

    # ── Base strategy matching ────────────────────────────────────────────────

    def _match_base_strategy(self, job: JobRequest) -> SchedulingStrategy:
        """
        Match the job to a base SchedulingStrategy using the routing table.

        Rules evaluated in priority order (most specific first):
          1. GPU_INFERENCE   — LC + GPU
          2. CPU_SERVING     — LC + no GPU
          3. GPU_TRAINING    — BATCH + GPU + not preemptible
          4. PREEMPTIBLE_ACTOR — any + preemptible
          5. STATEFUL_STREAM — STREAM + not preemptible
          6. GENERIC         — fallback (e.g. plain BATCH + no GPU)
        """
        wt = job.workload_type
        gpu = job.resources.gpu_required
        preemptible = job.preemptible

        # Rule 1: GPU Inference Server (e.g. vLLM, TorchServe with GPU)
        if wt == WorkloadType.LATENCY_CRITICAL and gpu:
            return self._strategy_gpu_inference()

        # Rule 2: CPU API / Serving (e.g. FastAPI, gRPC server)
        if wt == WorkloadType.LATENCY_CRITICAL and not gpu:
            return self._strategy_cpu_serving()

        # Rule 3: GPU Training / Learner (FSDP, DDP — stateful, don't interrupt)
        if wt == WorkloadType.BATCH and gpu and not preemptible:
            return self._strategy_gpu_training()

        # Rule 4: Preemptible actors (MCTS, ETL workers, ephemeral batch)
        if preemptible:
            return self._strategy_preemptible_actor()

        # Rule 5: Stateful stream (Kafka consumer, Reverb replay buffer)
        if wt == WorkloadType.STREAM and not preemptible:
            return self._strategy_stateful_stream()

        # Rule 6: Generic fallback (plain CPU batch, no special constraints)
        return self._strategy_generic()

    # ── Named strategy constructors ───────────────────────────────────────────

    @staticmethod
    def _strategy_gpu_inference() -> SchedulingStrategy:
        """
        GPU Inference Server — e.g. vLLM, TorchServe, TRT inference endpoint.

        Requirements:
          - GPU_NODE only (inference requires CUDA cores)
          - ON_DEMAND only (spot interruption = SLA breach for latency-critical)
          - Fast path (argmax η) — deterministic, no stochasticity
          - 30% CPU headroom (more than default 20% — inference is latency-sensitive)
          - Full spike penalty (spike → SLA miss)
          - Avoid nodes running BATCH training (GPU memory contention)

        Maps to: Ray Inference Server on TPU/GPU pod
        """
        return SchedulingStrategy(
            name="GPU_INFERENCE",
            required_arch=[NodeArch.GPU_NODE],
            required_instance=[InstanceType.ON_DEMAND],
            use_fast_path=True,
            allow_spot=False,
            sla_strict_threshold=0.30,
            spike_penalty_weight=1.0,
            spot_penalty_threshold=0.0,
            avoid_workload_types=[WorkloadType.BATCH],
        )

    @staticmethod
    def _strategy_cpu_serving() -> SchedulingStrategy:
        """
        CPU API / Serving — e.g. FastAPI, gRPC gateway, REST inference proxy.

        Requirements:
          - x86_64 or ARM64 (no GPU needed for CPU serving)
          - ON_DEMAND (SLA-bound, can't be interrupted)
          - Fast path (latency-critical → deterministic)
          - 25% headroom (slightly more than default; latency-sensitive)
          - High spike penalty (spike → increased tail latency)
          - No colocation restrictions (CPU serving doesn't contend with training)

        Maps to: Ray Actor serving CPU-based model endpoints
        """
        return SchedulingStrategy(
            name="CPU_SERVING",
            required_arch=[NodeArch.X86_64, NodeArch.ARM64],
            required_instance=[InstanceType.ON_DEMAND],
            use_fast_path=True,
            allow_spot=False,
            sla_strict_threshold=0.25,
            spike_penalty_weight=0.80,
            spot_penalty_threshold=0.0,
            avoid_workload_types=[],
        )

    @staticmethod
    def _strategy_gpu_training() -> SchedulingStrategy:
        """
        GPU Training / Learner — e.g. FSDP training, DDP, MuZero Learner.

        Requirements:
          - GPU_NODE (CUDA required for training)
          - Any instance type (spot OK — training checkpoints; can resume)
          - Full colony (stochastic — find best GPU packing, not just first fit)
          - Very low headroom requirement (throughput, not latency)
          - Low spike penalty (training throughput degrades gracefully)
          - Tolerates 50% interruption risk on spot (vs 30% default)
          - Can colocate with other training jobs

        Maps to: Ray Learner on TPU/GPU, PyTorch DDP Learner
        """
        return SchedulingStrategy(
            name="GPU_TRAINING",
            required_arch=[NodeArch.GPU_NODE],
            required_instance=None,
            use_fast_path=False,
            allow_spot=True,
            sla_strict_threshold=0.05,
            spike_penalty_weight=0.20,
            spot_penalty_threshold=0.50,
            avoid_workload_types=[],
        )

    @staticmethod
    def _strategy_preemptible_actor() -> SchedulingStrategy:
        """
        Preemptible Actor — e.g. MCTS actor, ETL worker, data loader.

        Requirements:
          - x86_64 or ARM64 (CPU-bound simulation/data work)
          - Any instance type (spot strongly preferred — cheap, ephemeral)
          - Full colony (cost-optimal packing matters for many actors)
          - Minimal headroom requirement (actors handle interruption gracefully)
          - Very low spike penalty (don't care about CPU spikes)
          - Very high spot tolerance (70% interruption probability acceptable)
          - No colocation restrictions

        Maps to: Ray Actor for environment simulation, MCTS, data pipeline
        """
        return SchedulingStrategy(
            name="PREEMPTIBLE_ACTOR",
            required_arch=[NodeArch.X86_64, NodeArch.ARM64],
            required_instance=None,
            use_fast_path=False,
            allow_spot=True,
            sla_strict_threshold=0.05,
            spike_penalty_weight=0.10,
            spot_penalty_threshold=0.70,
            avoid_workload_types=[],
        )

    @staticmethod
    def _strategy_stateful_stream() -> SchedulingStrategy:
        """
        Stateful Stream — e.g. Kafka consumer, Reverb replay buffer, feature store.

        Requirements:
          - Any arch (memory-bound, not compute-specific)
          - ON_DEMAND only (stateful — data loss on interruption)
          - Full colony (find best memory-fit node)
          - Moderate headroom (needs room for memory spikes)
          - Moderate spike penalty (memory spikes cause backpressure)
          - No spot tolerance (stateful data must not be lost)

        Maps to: Ray Object Store, Reverb Server, Kafka consumer group
        """
        return SchedulingStrategy(
            name="STATEFUL_STREAM",
            required_arch=None,
            required_instance=[InstanceType.ON_DEMAND],
            use_fast_path=False,
            allow_spot=False,
            sla_strict_threshold=0.15,
            spike_penalty_weight=0.40,
            spot_penalty_threshold=0.0,
            avoid_workload_types=[],
        )

    @staticmethod
    def _strategy_generic() -> SchedulingStrategy:
        """
        Generic fallback — plain CPU batch jobs, simple workloads.

        All thresholds at CostEngine defaults. Full colony run.
        Spot allowed (cost-efficient for non-critical batch).
        No colocation restrictions.

        Maps to: Any workload not matching a specific ML role.
        """
        return SchedulingStrategy(
            name="GENERIC",
            required_arch=None,
            required_instance=None,
            use_fast_path=False,
            allow_spot=True,
            sla_strict_threshold=0.20,
            spike_penalty_weight=0.50,
            spot_penalty_threshold=0.30,
            avoid_workload_types=[],
        )

    # ── Deadline-urgent override ──────────────────────────────────────────────

    @staticmethod
    def _apply_deadline_override(
        job: JobRequest,
        strategy: SchedulingStrategy,
    ) -> None:
        """
        If the job's deadline is imminent (< DEADLINE_URGENT_WINDOW_S seconds away),
        tighten the strategy in place.

        Changes applied:
          - use_fast_path = True      (argmax, skip colony stochasticity)
          - sla_strict_threshold += SLA_DEADLINE_BOOST  (tighter headroom)
          - avoid_workload_types extended to include BATCH + STREAM
            (avoid all noisy neighbours when deadline is critical)

        Why modify in place rather than return new object?
            The base strategy is already correct for this workload class.
            The deadline override is an urgency modifier, not a class change.
            Modifying in place keeps the strategy.name intact for logging.

        Args:
            job:      The JobRequest being classified.
            strategy: The base SchedulingStrategy (mutated in place if urgent).
        """
        if job.deadline_epoch is None:
            return

        time_to_deadline = job.deadline_epoch - time.time()
        if time_to_deadline >= DEADLINE_URGENT_WINDOW_S:
            return

        # Deadline is imminent — tighten everything
        strategy.use_fast_path = True
        strategy.sla_strict_threshold = min(
            strategy.sla_strict_threshold + SLA_DEADLINE_BOOST,
            0.90,   # cap at 90% — don't make placement impossible
        )

        # Avoid all noisy neighbours
        noisy_neighbours = [WorkloadType.BATCH, WorkloadType.STREAM]
        for wt in noisy_neighbours:
            if wt not in strategy.avoid_workload_types:
                strategy.avoid_workload_types.append(wt)

    def __repr__(self) -> str:
        return "WorkloadIntentRouter(rules=6)"
