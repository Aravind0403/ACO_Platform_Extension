"""
orchestrator/control_plane/orchestration_service.py
─────────────────────────────────────────────────────
V2 OrchestratorService: the central control plane state machine.

What changed from V1
─────────────────────
V1 OrchestratorService:
  - Called naive_schedule() directly
  - No telemetry ingestion
  - No predictor integration
  - No WorkloadProfile tracking
  - No cost awareness

V2 adds:
  1. aco_schedule() as primary scheduler (naive_schedule() kept as fallback)
  2. WorkloadProfile registry — updated on every job completion
  3. WorkloadPredictor registry — one predictor per node, refitted every 60s
     via refit_all_predictors() (called by telemetry/collector.py in Phase 7)
  4. Telemetry ingestion — update_node_telemetry() updates node.latest_telemetry
     so CostEngine.sla_headroom_factor() gets real CPU utilisation data
  5. Richer mock cluster — 5 nodes across CPU/ARM64/GPU tiers with cost profiles

What is NOT changed
────────────────────
  - _allocate_resources() / _release_resources() — identical to V1
  - complete_job() — identical to V1 (+ WorkloadProfile update)
  - get_job_status() / get_active_jobs() / get_completed_jobs() — identical
  - SchedulingFailedError / AdmissionRejectedError handling — identical

Thread safety
──────────────
Not thread-safe. The service runs on the asyncio event loop (single thread).
All state mutations (node_state, active_jobs, etc.) happen synchronously.
If concurrent requests are needed in Phase 9, add asyncio.Lock around
submit_job() and complete_job().
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.shared.models import (
    ComputeNode,
    InstanceType,
    JobExecution,
    JobRequest,
    JobState,
    NodeArch,
    NodeCostProfile,
    NodeState,
    NodeTelemetry,
    PredictionResult,
    ResourceRequest,
    WorkloadType,
)
from orchestrator.shared.telemetry import ResourceSample, WorkloadProfile
from orchestrator.control_plane.admission_controller import (
    admit_job,
    AdmissionRejectedError,
)
from orchestrator.control_plane.scheduler import (
    SchedulingFailedError,
    aco_schedule,
    naive_schedule,
)
from orchestrator.control_plane.predictor import WorkloadPredictor
from orchestrator.control_plane.intent_router import WorkloadIntentRouter

logger = logging.getLogger(__name__)


# ── Back-pressure queue configuration ────────────────────────────────────────
QUEUE_MAX_DEPTH: int = 50
"""Maximum number of jobs held in the pending queue when cluster is saturated."""

QUEUE_TTL_S: float = 60.0
"""Time-to-live for queued jobs. Jobs older than this are expired on drain."""

# ── Cross-call pheromone update parameters ────────────────────────────────────
_PHEROMONE_RHO: float = 0.1
"""Evaporation rate applied to all nodes after each scheduling decision."""

_PHEROMONE_Q: float = 1.0
"""Deposit amount added to the chosen node's pheromone level."""


class OrchestratorService:
    """
    Central control plane: state management, admission, scheduling, telemetry.

    V2 public API (backwards compatible with V1):
        submit_job(request_data)           → Dict[str, str]
        complete_job(job_id, success)      → Dict[str, str]
        get_job_status(job_id)             → Optional[JobExecution]
        get_active_jobs()                  → List[JobExecution]
        get_completed_jobs(limit)          → List[JobExecution]

    V2 additions:
        update_node_telemetry(telemetry)   → None  (called by data plane agent)
        get_prediction(node_id)            → Optional[PredictionResult]
        refit_all_predictors()             → None  (called by background loop)
        get_scheduling_metrics()           → Dict  (for /metrics endpoint)

    Attributes:
        node_state         : Dict[str, ComputeNode]       — live node registry
        active_jobs        : Dict[str, JobExecution]      — in-flight jobs
        completed_jobs     : deque(maxlen=100)            — job history
        workload_profiles  : Dict[str, WorkloadProfile]   — per-workload-type stats
        predictors         : Dict[str, WorkloadPredictor] — per-node LSTM predictors
        scheduling_latencies: deque(maxlen=1000)          — for P99 metric
    """

    def __init__(self) -> None:
        # ── Core state ────────────────────────────────────────────────────────
        self.node_state: Dict[str, ComputeNode] = {}
        self.active_jobs: Dict[str, JobExecution] = {}
        self.completed_jobs: deque = deque(maxlen=100)
        self.scheduling_latencies: deque = deque(maxlen=1000)

        # ── V2 additions ──────────────────────────────────────────────────────
        # One WorkloadProfile per workload type — accumulates ResourceSamples
        # from completed jobs. Feeds the LSTM predictors.
        self.workload_profiles: Dict[str, WorkloadProfile] = {
            wt.value: WorkloadProfile(workload_name=wt.value)
            for wt in WorkloadType
        }

        # One WorkloadPredictor per node — independently trained on that node's
        # completed-job CPU history.
        self.predictors: Dict[str, WorkloadPredictor] = {}

        # Cache of latest predictions (populated by refit_all_predictors)
        self._prediction_cache: Dict[str, PredictionResult] = {}

        # ── Phase 6 additions ─────────────────────────────────────────────────
        # Tracks which WorkloadTypes are currently running on each node.
        # Used by aco_schedule() for the colocation policy (avoid_workload_types).
        # Updated by _allocate_resources() and _release_resources().
        self._node_workload_map: Dict[str, List[WorkloadType]] = defaultdict(list)

        # Shared WorkloadIntentRouter — stateless, one instance is fine.
        self._router = WorkloadIntentRouter()

        # ── Phase 10: Back-pressure queue ─────────────────────────────────────
        # When all nodes are saturated, jobs are held in this bounded queue
        # instead of being silently dropped. Each entry is:
        #   (request_data: dict, enqueued_at: float, scheduling_latency_ms: Optional[float])
        # drain_pending_queue() retries placement for each entry.
        # Jobs older than QUEUE_TTL_S are expired (EXPIRED status).
        self._pending_queue: deque = deque(maxlen=QUEUE_MAX_DEPTH)

        # ── Phase 10: Cross-call pheromone persistence ────────────────────────
        # One float per node — the accumulated pheromone from all past scheduling
        # calls. Starts at TAU_INITIAL (1.0) = no prior. After each successful
        # placement the chosen node's level is deposited and all nodes evaporate.
        # Passed to aco_schedule() as node_pheromone so the colony starts informed
        # by history rather than blind on every call.
        # Can be serialised/restored via save_pheromone_snapshot() /
        # load_pheromone_snapshot() to survive API process restarts.
        self._node_pheromone: Dict[str, float] = {}   # populated after cluster init

        # ── Thread safety ─────────────────────────────────────────────────────
        # RLock (reentrant) because drain_pending_queue() calls submit_job()
        # internally — a plain Lock would deadlock on that second acquisition.
        # Protects: node_state, active_jobs, completed_jobs, _pending_queue,
        #           _node_pheromone, _node_workload_map, _prediction_cache.
        # NOT held during LSTM training (refit_all_predictors splits around it).
        self._state_lock = threading.RLock()

        # ── Bootstrap cluster ─────────────────────────────────────────────────
        self._initialize_mock_nodes()
        logger.info(
            "OrchestratorService (V2) initialised with %d nodes.",
            len(self.node_state),
        )

    # ── Cluster initialisation ─────────────────────────────────────────────────

    def _initialize_mock_nodes(self) -> None:
        """
        Bootstrap a representative 5-node cluster.

        Node mix mirrors a real heterogeneous cluster:
          node-cpu-01  : x86_64 on-demand, high CPU/mem. BATCH & STREAM.
          node-arm-02  : ARM64 spot, cheaper per core. Cost-efficient for BATCH.
          node-api-03  : x86_64 on-demand, small. Dedicated LATENCY_CRITICAL.
          node-gpu-04  : GPU node, on-demand. ML training (BATCH).
          node-gpu-05  : GPU node, spot (cheaper). Interruptible ML training.

        V1 had 3 nodes. V2 extends to 5 for richer scheduler coverage.
        All cost_profile values are illustrative (not real AWS prices).
        """
        nodes = [
            ComputeNode(
                node_id="node-cpu-01",
                arch=NodeArch.X86_64,
                total_cpu_cores=32.0,
                total_memory_gb=128.0,
                cost_profile=NodeCostProfile(
                    instance_type=InstanceType.ON_DEMAND,
                    cost_per_hour_usd=0.48,
                    region="us-east-1",
                ),
            ),
            ComputeNode(
                node_id="node-arm-02",
                arch=NodeArch.ARM64,
                total_cpu_cores=32.0,
                total_memory_gb=64.0,
                cost_profile=NodeCostProfile(
                    instance_type=InstanceType.SPOT,
                    cost_per_hour_usd=0.12,
                    interruption_prob=0.15,
                    region="us-east-1",
                ),
            ),
            ComputeNode(
                node_id="node-api-03",
                arch=NodeArch.X86_64,
                total_cpu_cores=8.0,
                total_memory_gb=16.0,
                cost_profile=NodeCostProfile(
                    instance_type=InstanceType.ON_DEMAND,
                    cost_per_hour_usd=0.12,
                    region="us-east-1",
                ),
            ),
            ComputeNode(
                node_id="node-gpu-04",
                arch=NodeArch.GPU_NODE,
                total_cpu_cores=16.0,
                total_memory_gb=64.0,
                gpu_inventory={"A100": 4},
                gpu_vram_gb={"A100": 80.0},
                cost_profile=NodeCostProfile(
                    instance_type=InstanceType.ON_DEMAND,
                    cost_per_hour_usd=3.20,
                    region="us-east-1",
                ),
            ),
            ComputeNode(
                node_id="node-gpu-05",
                arch=NodeArch.GPU_NODE,
                total_cpu_cores=16.0,
                total_memory_gb=64.0,
                gpu_inventory={"A100": 4},
                gpu_vram_gb={"A100": 80.0},
                cost_profile=NodeCostProfile(
                    instance_type=InstanceType.SPOT,
                    cost_per_hour_usd=0.96,
                    interruption_prob=0.20,
                    region="us-east-1",
                ),
            ),
        ]
        for node in nodes:
            self.node_state[node.node_id] = node
            # Create a predictor for each node upfront (untrained until data arrives)
            self.predictors[node.node_id] = WorkloadPredictor(node_id=node.node_id)
            # Initialise cross-call pheromone to TAU_INITIAL (1.0 = no prior knowledge)
            self._node_pheromone[node.node_id] = 1.0

    # ── Primary public API ─────────────────────────────────────────────────────

    def submit_job(
        self,
        request_data: dict,
        scheduling_latency_ms: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Receive a job submission, run the full control plane pipeline.

        Pipeline (same structure as V1, upgraded internals):
          1. Pydantic validation (JobRequest(**request_data))
          2. Admission control (admit_job)
          3. ACO scheduling (aco_schedule with CostEngine η)
             → falls back to naive_schedule on ColonyFailedError
          4. Resource allocation (_allocate_resources)
          5. JobExecution record creation

        Args:
            request_data:          Raw dict from API layer.
            scheduling_latency_ms: Measured latency from API layer (optional).

        Returns:
            {"status": "SCHEDULED"|"REJECTED"|"ERROR", "job_id", "node_id", "message"}
        """
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        request_data["job_id"] = job_id
        submission_time = datetime.now(timezone.utc)

        self._state_lock.acquire()
        try:
            # 1. Schema validation
            job_request = JobRequest(**request_data)

            # 2. Admission control
            admit_job(job_request)

            # 3. Intent classification + ACO scheduling
            strategy = self._router.classify(job_request)
            available_nodes = list(self.node_state.values())
            selected_node_id = aco_schedule(
                job_request=job_request,
                available_nodes=available_nodes,
                predictors=self._prediction_cache,
                strategy=strategy,
                node_workload_map=dict(self._node_workload_map),
                node_pheromone=dict(self._node_pheromone),
            )
            # ── Update cross-call pheromone ────────────────────────────────
            # Evaporate all nodes, then deposit on the chosen node.
            # Deposit is a fixed Q per successful placement so nodes selected
            # more often naturally accumulate higher pheromone (standard ACO
            # elitist deposit without needing the path cost here).
            for nid in self._node_pheromone:
                self._node_pheromone[nid] = max(
                    0.01, min(10.0, self._node_pheromone[nid] * (1.0 - _PHEROMONE_RHO))
                )
            self._node_pheromone[selected_node_id] = min(
                10.0,
                self._node_pheromone[selected_node_id] + _PHEROMONE_Q,
            )

            # 4. Resource allocation
            self._allocate_resources(selected_node_id, job_request)

            # 5. Job execution record
            scheduled_time = datetime.now(timezone.utc)
            job_execution = JobExecution(
                job_id=job_id,
                job_request=job_request,
                assigned_node_id=selected_node_id,
                state=JobState.RUNNING,
                submitted_at=submission_time,
                scheduled_at=scheduled_time,
                started_at=scheduled_time,
                scheduling_latency_ms=scheduling_latency_ms or 0.0,
            )
            self.active_jobs[job_id] = job_execution

            if scheduling_latency_ms is not None:
                self.scheduling_latencies.append(scheduling_latency_ms)

            logger.info("Job %s scheduled → node %s", job_id, selected_node_id)
            return {
                "status": "SCHEDULED",
                "job_id": job_id,
                "node_id": selected_node_id,
                "message": f"Job placed on {selected_node_id} via ACO scheduler",
            }

        except AdmissionRejectedError as e:
            # Admission failures are permanent (invalid job spec) — reject immediately.
            return {
                "status": "REJECTED",
                "job_id": job_id,
                "node_id": None,
                "message": str(e),
            }
        except SchedulingFailedError as e:
            # Distinguish permanent vs transient failure:
            #   Permanent  — no node has enough TOTAL resources (can never fit even empty).
            #                E.g., job requires 9999 CPU cores but max node has 32.
            #   Transient  — nodes could fit the job if they weren't currently allocated.
            #                E.g., all 32-core nodes are busy with other jobs.
            # Only transient failures are worth queuing.
            could_ever_fit = any(
                n.total_cpu_cores >= job_request.resources.cpu_cores_min
                and n.total_memory_gb >= job_request.resources.memory_gb_min
                and (not job_request.resources.gpu_required or bool(n.gpu_inventory))
                and n.state == NodeState.HEALTHY
                for n in self.node_state.values()
            )
            if not could_ever_fit:
                return {
                    "status": "REJECTED",
                    "job_id": job_id,
                    "node_id": None,
                    "message": str(e),
                }
            # Transient failure — queue the job for retry.
            if len(self._pending_queue) < QUEUE_MAX_DEPTH:
                self._pending_queue.append((request_data, time.monotonic(), scheduling_latency_ms))
                position = len(self._pending_queue)
                logger.info(
                    "Job %s queued (cluster saturated, queue depth=%d).", job_id, position
                )
                return {
                    "status": "QUEUED",
                    "job_id": job_id,
                    "node_id": None,
                    "queue_position": position,
                    "message": (
                        f"Cluster saturated — job queued at position {position}. "
                        f"Will retry within {int(QUEUE_TTL_S)}s."
                    ),
                }
            else:
                # Queue is full — hard reject.
                return {
                    "status": "REJECTED",
                    "job_id": job_id,
                    "node_id": None,
                    "message": f"Cluster saturated and queue full ({QUEUE_MAX_DEPTH} jobs). {e}",
                }
        except Exception as e:
            logger.exception("Unexpected error in submit_job for %s", job_id)
            return {
                "status": "ERROR",
                "job_id": job_id,
                "node_id": None,
                "message": f"Unexpected error: {e.__class__.__name__}: {e}",
            }
        finally:
            self._state_lock.release()

    def drain_pending_queue(self) -> List[Dict[str, Any]]:
        """
        Retry placement for all queued jobs. Called periodically by the API's
        background loop (every _TELEMETRY_INTERVAL_S seconds after collector.tick).

        For each queued entry:
          - If older than QUEUE_TTL_S → expire it (status=EXPIRED).
          - Otherwise → call submit_job() to retry placement.
            If SCHEDULED → remove from queue (drain successful).
            If QUEUED again → stop (cluster still saturated, no point continuing).
            If REJECTED → remove from queue (permanent failure).

        Returns:
            List of outcome dicts for monitoring/logging (one per entry processed).
        """
        with self._state_lock:
            return self._drain_pending_queue_locked()

    def _drain_pending_queue_locked(self) -> List[Dict[str, Any]]:
        """Lock-held implementation — called only from drain_pending_queue()."""
        outcomes: List[Dict[str, Any]] = []
        now = time.monotonic()

        # Process queue front-to-back; stop early if cluster still saturated
        entries_to_retry = list(self._pending_queue)
        self._pending_queue.clear()

        for request_data, enqueued_at, lat_ms in entries_to_retry:
            age_s = now - enqueued_at
            if age_s > QUEUE_TTL_S:
                outcomes.append({
                    "status": "EXPIRED",
                    "job_id": request_data.get("job_id"),
                    "age_s": round(age_s, 1),
                })
                logger.warning(
                    "Queued job %s expired after %.1fs (TTL=%ds).",
                    request_data.get("job_id"), age_s, int(QUEUE_TTL_S),
                )
                continue

            result = self.submit_job(request_data, scheduling_latency_ms=lat_ms)
            outcomes.append(result)

            if result["status"] == "QUEUED":
                # Cluster still saturated — re-queue the remaining entries and stop
                self._pending_queue.append((request_data, enqueued_at, lat_ms))
                # Re-queue remaining (we already cleared the deque above)
                for remaining in entries_to_retry[entries_to_retry.index((request_data, enqueued_at, lat_ms)) + 1:]:
                    self._pending_queue.append(remaining)
                break

        return outcomes

    def get_queue_status(self) -> Dict[str, Any]:
        """Return current queue depth and oldest job age for the API /queue endpoint."""
        now = time.monotonic()
        if not self._pending_queue:
            return {"depth": 0, "oldest_age_s": 0.0, "max_depth": QUEUE_MAX_DEPTH, "ttl_s": QUEUE_TTL_S}
        oldest_age = now - min(entry[1] for entry in self._pending_queue)
        return {
            "depth": len(self._pending_queue),
            "oldest_age_s": round(oldest_age, 1),
            "max_depth": QUEUE_MAX_DEPTH,
            "ttl_s": QUEUE_TTL_S,
        }

    def complete_job(
        self,
        job_id: str,
        success: bool = True,
        failure_reason: Optional[str] = None,
        actual_cpu_used_cores: Optional[float] = None,
        actual_memory_used_gb: Optional[float] = None,
        actual_scheduling_latency_ms: Optional[float] = None,
    ) -> Dict[str, str]:
        """
        Mark a job complete, release resources, update WorkloadProfile.

        V2 addition: When actual resource usage is reported, add a
        ResourceSample to the appropriate WorkloadProfile. This feeds
        the LSTM predictor — the more jobs complete, the better predictions get.

        Args:
            job_id:                     The job to complete.
            success:                    True = COMPLETED, False = FAILED.
            failure_reason:             Optional failure message.
            actual_cpu_used_cores:      Actual CPU peak usage (from node agent).
            actual_memory_used_gb:      Actual memory peak usage.
            actual_scheduling_latency_ms: Actual end-to-end scheduling latency.

        Returns:
            {"status": "SUCCESS"|"ERROR", ...}
        """
        if job_id not in self.active_jobs:
            # Check if already completed
            for completed_job in self.completed_jobs:
                if completed_job.job_id == job_id:
                    return {
                        "status": "ERROR",
                        "message": f"Job {job_id} already completed at {completed_job.completed_at}",
                    }
            return {"status": "ERROR", "message": f"Job {job_id} not found"}

        job_execution = self.active_jobs[job_id]
        job_execution.state = JobState.COMPLETED if success else JobState.FAILED
        job_execution.completed_at = datetime.now(timezone.utc)
        if failure_reason:
            job_execution.failure_reason = failure_reason

        # Release physical resources
        self._release_resources(job_execution.assigned_node_id, job_execution.job_request)

        # V2: Update WorkloadProfile if resource usage was reported
        if actual_cpu_used_cores is not None:
            self._update_workload_profile(
                job_execution=job_execution,
                cpu_cores_used=actual_cpu_used_cores,
                memory_gb_used=actual_memory_used_gb or job_execution.job_request.resources.memory_gb_min,
                scheduling_latency_ms=actual_scheduling_latency_ms or 0.0,
            )

        del self.active_jobs[job_id]
        self.completed_jobs.append(job_execution)

        runtime = (
            (job_execution.completed_at - job_execution.started_at).total_seconds()
            if job_execution.started_at else 0.0
        )
        logger.info(
            "Job %s → %s (runtime: %.2fs)",
            job_id, job_execution.state.value, runtime,
        )

        return {
            "status": "SUCCESS",
            "job_id": job_id,
            "final_state": job_execution.state.value,
            "message": f"Job completed and resources released from {job_execution.assigned_node_id}",
        }

    # ── Read-only queries (unchanged from V1) ──────────────────────────────────

    def get_job_status(self, job_id: str) -> Optional[JobExecution]:
        """Return job execution record (active or completed). None if not found."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return job
        return None

    def get_active_jobs(self) -> List[JobExecution]:
        """Return all currently running/scheduled jobs."""
        return list(self.active_jobs.values())

    def get_completed_jobs(self, limit: int = 50) -> List[JobExecution]:
        """Return recently completed jobs (newest first, up to limit)."""
        return list(self.completed_jobs)[:limit]

    # ── V2 additions: telemetry + prediction ──────────────────────────────────

    def update_node_telemetry(self, telemetry: NodeTelemetry) -> None:
        """
        Accept a live heartbeat from a node agent. Updates node.latest_telemetry
        so CostEngine.sla_headroom_factor() uses real CPU utilisation data
        instead of the allocation-based estimate.

        Called by: telemetry/collector.py (Phase 7 background loop).

        Args:
            telemetry: NodeTelemetry snapshot from the node agent.
        """
        with self._state_lock:
            node = self.node_state.get(telemetry.node_id)
            if node is None:
                logger.warning(
                    "update_node_telemetry: unknown node_id=%s", telemetry.node_id
                )
                return
            node.latest_telemetry = telemetry
        logger.debug(
            "Telemetry updated: node=%s cpu=%.1f%% mem=%.1f%%",
            telemetry.node_id, telemetry.cpu_util_pct, telemetry.memory_util_pct,
        )

    def get_prediction(self, node_id: str) -> Optional[PredictionResult]:
        """
        Return the latest cached prediction for a node.

        Returns None if the node has no predictor or no prediction yet.
        Used by GET /predict/{node_id} (Phase 9 API).
        """
        return self._prediction_cache.get(node_id)

    def refit_all_predictors(self) -> None:
        """
        Refit all node predictors if enough new samples have accumulated.

        Call this from the background telemetry loop every 60 seconds.
        Each predictor checks internally whether it needs a refit
        (only triggers if ≥10 new samples since last fit).

        After refitting, updates the prediction cache so aco_schedule()
        gets fresh CostEngine hints on the next call.

        Note: WorkloadProfile is keyed by workload type, but predictors
        are keyed by node_id. For Phase 5, we use the BATCH profile as a
        proxy for all node predictors (assumes mixed-workload cluster).
        In Phase 7, we'll track per-node resource histories separately.
        """
        # Use BATCH workload profile as the representative signal
        # (most clusters are primarily batch-heavy)
        for node_id, predictor in self.predictors.items():
            # Find the workload profile with the most samples
            best_profile = max(
                self.workload_profiles.values(),
                key=lambda p: len(p.samples),
                default=None,
            )
            if best_profile is None or len(best_profile.samples) == 0:
                continue

            predictor.refit_if_needed(best_profile)

            # Predict outside the lock (may run LSTM inference, ~1ms)
            node = self.node_state.get(node_id)
            if node:
                result = predictor.predict(best_profile)
                # Lock only for the cache write — keeps training non-blocking
                with self._state_lock:
                    self._prediction_cache[node_id] = PredictionResult(
                        node_id=node_id,
                        forecast_horizon_min=result.forecast_horizon_min,
                        predicted_cpu_util=result.predicted_cpu_util,
                        predicted_memory_util=result.predicted_memory_util,
                        predicted_gpu_util=result.predicted_gpu_util,
                        spike_probability=result.spike_probability,
                        confidence=result.confidence,
                    )

    def get_scheduling_metrics(self) -> dict:
        """
        Return current scheduling metrics for the /metrics endpoint.

        Metrics:
            active_jobs:          Count of in-flight jobs.
            completed_jobs:       Count of completed/failed jobs in history.
            scheduling_p99_ms:    P99 scheduling latency across recent jobs.
            avg_scheduling_ms:    Mean scheduling latency.
            node_utilisation:     Per-node CPU utilisation %.
            profile_sample_counts: Number of samples per workload profile.
        """
        latencies = list(self.scheduling_latencies)
        if latencies:
            sorted_lat = sorted(latencies)
            p99_idx = max(0, int(0.99 * len(sorted_lat)) - 1)
            p99 = sorted_lat[p99_idx]
            avg = sum(latencies) / len(latencies)
        else:
            p99 = 0.0
            avg = 0.0

        return {
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "scheduling_p99_ms": round(p99, 2),
            "avg_scheduling_ms": round(avg, 2),
            "node_utilisation": {
                node_id: round(node.cpu_utilisation_pct, 1)
                for node_id, node in self.node_state.items()
            },
            "profile_sample_counts": {
                name: len(profile.samples)
                for name, profile in self.workload_profiles.items()
            },
        }

    # ── Pheromone snapshot (cross-restart persistence) ──────────────────────────

    def save_pheromone_snapshot(self, path: str = "/tmp/aco_pheromone.json") -> None:
        """
        Persist the current per-node pheromone vector to a JSON file.

        Called on API shutdown so learned placement preferences survive restarts.
        The file is small (~100 bytes for 5 nodes) and written atomically via
        a temp-and-rename pattern to avoid partial-write corruption.
        """
        import json, os, tempfile
        data = {"version": 1, "pheromone": self._node_pheromone}
        dir_ = os.path.dirname(path) or "."
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=dir_, suffix=".tmp", delete=False
            ) as tmp:
                json.dump(data, tmp, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, path)
            logger.info("Pheromone snapshot saved to %s (%d nodes)", path, len(self._node_pheromone))
        except OSError as exc:
            logger.warning("Could not save pheromone snapshot to %s: %s", path, exc)

    def load_pheromone_snapshot(self, path: str = "/tmp/aco_pheromone.json") -> None:
        """
        Restore per-node pheromone from a previously saved JSON snapshot.

        Called on API startup. Only restores values for nodes that exist in
        the current cluster (safe against node additions/removals between restarts).
        Missing nodes are left at TAU_INITIAL=1.0.
        """
        import json, os
        if not os.path.exists(path):
            logger.info("No pheromone snapshot at %s — starting fresh.", path)
            return
        try:
            with open(path) as f:
                data = json.load(f)
            saved = data.get("pheromone", {})
            restored = 0
            for node_id in self._node_pheromone:
                if node_id in saved:
                    self._node_pheromone[node_id] = float(saved[node_id])
                    restored += 1
            logger.info(
                "Pheromone snapshot loaded from %s (%d/%d nodes restored).",
                path, restored, len(self._node_pheromone),
            )
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("Could not load pheromone snapshot from %s: %s", path, exc)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _allocate_resources(
        self, node_id: str, job_request: JobRequest
    ) -> None:
        """
        Update node state to reflect new job allocation. Identical to V1.
        Phase 6: also records the job's workload type in _node_workload_map.
        """
        node = self.node_state.get(node_id)
        if not node:
            return

        resources = job_request.resources
        node.allocated_cpu_cores += resources.cpu_cores_min
        node.allocated_memory_gb += resources.memory_gb_min

        if resources.gpu_required:
            # Allocate from the first GPU model that has capacity
            for model in node.gpu_inventory:
                if node.available_gpus(model) >= resources.gpu_count:
                    node.allocated_gpus[model] = (
                        node.allocated_gpus.get(model, 0) + resources.gpu_count
                    )
                    break

        # Phase 6: track workload type on this node for colocation policy
        self._node_workload_map[node_id].append(job_request.workload_type)

        logger.debug(
            "Allocated: node=%s cpu=%.1f mem=%.1fGB",
            node_id,
            resources.cpu_cores_min,
            resources.memory_gb_min,
        )

    def _release_resources(
        self, node_id: str, job_request: JobRequest
    ) -> None:
        """
        Release resources when a job completes. Identical to V1.
        Phase 6: also removes the job's workload type from _node_workload_map.
        """
        node = self.node_state.get(node_id)
        if not node:
            logger.warning(
                "_release_resources: node %s not found", node_id
            )
            return

        resources = job_request.resources
        node.allocated_cpu_cores = max(
            0.0, node.allocated_cpu_cores - resources.cpu_cores_min
        )
        node.allocated_memory_gb = max(
            0.0, node.allocated_memory_gb - resources.memory_gb_min
        )

        if resources.gpu_required:
            for model in list(node.allocated_gpus.keys()):
                if node.allocated_gpus[model] > 0:
                    node.allocated_gpus[model] = max(
                        0, node.allocated_gpus[model] - resources.gpu_count
                    )
                    if node.allocated_gpus[model] == 0:
                        del node.allocated_gpus[model]
                    break

        # Phase 6: remove workload type from colocation map (remove first occurrence)
        wt_list = self._node_workload_map.get(node_id)
        if wt_list and job_request.workload_type in wt_list:
            wt_list.remove(job_request.workload_type)
            if not wt_list:
                del self._node_workload_map[node_id]

        logger.debug(
            "Released: node=%s  avail_cpu=%.1f  avail_mem=%.1fGB",
            node_id,
            node.available_cpu_cores,
            node.available_memory_gb,
        )

    def _update_workload_profile(
        self,
        job_execution: JobExecution,
        cpu_cores_used: float,
        memory_gb_used: float,
        scheduling_latency_ms: float,
    ) -> None:
        """
        Add a ResourceSample to the appropriate WorkloadProfile.

        Called by complete_job() when actual resource usage is known.
        Over time, this feeds the LSTM predictor — more data = better forecasts.

        Args:
            job_execution:         The completed job.
            cpu_cores_used:        Actual peak CPU cores consumed.
            memory_gb_used:        Actual peak memory consumed.
            scheduling_latency_ms: End-to-end scheduling latency for this job.
        """
        workload_key = job_execution.job_request.workload_type.value
        profile = self.workload_profiles.get(workload_key)
        if profile is None:
            return

        # Compute actual duration
        duration_s = 0.0
        if job_execution.started_at and job_execution.completed_at:
            duration_s = (
                job_execution.completed_at - job_execution.started_at
            ).total_seconds()

        sample = ResourceSample(
            cpu_cores_used=max(0.0, cpu_cores_used),
            memory_gb_used=max(0.0, memory_gb_used),
            duration_s=max(0.0, duration_s),
            scheduling_latency_ms=max(0.0, scheduling_latency_ms),
        )
        profile.add_sample(sample)

        # Track scheduling latency for metrics
        if scheduling_latency_ms > 0:
            self.scheduling_latencies.append(scheduling_latency_ms)

        logger.debug(
            "WorkloadProfile[%s] updated: n=%d  avg_cpu=%.2f  p99_latency=%.2fms",
            workload_key, len(profile.samples),
            profile.avg_cpu_cores, profile.p99_latency_ms,
        )
