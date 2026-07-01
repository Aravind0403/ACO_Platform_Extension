"""
orchestrator/shared/models.py
─────────────────────────────
The single source of truth for every data structure in ACO V2.

Design philosophy
-----------------
Every model answers one question: "What does the system *need to know*
about this thing in order to make a smart scheduling decision?"

V1 kept things simple (and correct). V2 extends each model with the
fields that the ACO core, the cost engine, and the predictor will need.
No V1 field has been removed — all callers still work as-is.

Reading guide
-------------
Read top-to-bottom. Each model builds on the ones above it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: ENUMERATIONS
# Small named constants that make the code self-documenting.
# ─────────────────────────────────────────────────────────────────────────────

class WorkloadType(str, Enum):
    """
    The three classes of work the scheduler recognises.

    Why it matters:
        Each class has different SLA constraints and hardware affinity.
        The admission controller enforces different rules per class.
        The ACO heuristic weights change per class.

    LATENCY_CRITICAL  → API servers, inference endpoints. Needs <10ms scheduling.
    BATCH             → ML training, ETL. Throughput over latency. GPU-heavy.
    STREAM            → Kafka consumers, real-time pipelines. Bounded throughput.
    """
    LATENCY_CRITICAL = "latency-critical"
    BATCH = "batch"
    STREAM = "stream-processing"


class NodeState(str, Enum):
    """
    Operational lifecycle of a compute node.

    HEALTHY     → Accepting new jobs. Full capacity.
    DEGRADED    → Running but with reduced capacity (e.g., a failing disk,
                  thermal throttle). Scheduler deprioritises these nodes.
    MAINTENANCE → Manually cordoned. No new jobs. Drain in progress.
    DRAINING    → System-initiated drain (e.g., spot reclaim in 2 min).
                  Existing jobs continue; no new placements.
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    DRAINING = "draining"


class NodeArch(str, Enum):
    """
    Hardware architecture of a compute node.

    V1 did not track this. V2 needs it because:
    - Some jobs require AVX-512 (x86_64 only).
    - ARM64 nodes (e.g., AWS Graviton) are cheaper per core.
    - GPU nodes have a separate scheduling path entirely.

    The ACO heuristic uses arch affinity as part of η (eta).
    """
    X86_64 = "x86_64"
    ARM64 = "arm64"
    GPU_NODE = "gpu-node"   # x86_64 host with attached NVIDIA GPUs


class InstanceType(str, Enum):
    """
    Billing model for a compute node.

    ON_DEMAND → Reserved capacity. Predictable. Expensive.
    SPOT      → Spare capacity. 70-90% cheaper. Can be reclaimed
                with 2-minute warning (AWS) or 30-second (GCP).

    The cost engine penalises SPOT nodes for LATENCY_CRITICAL jobs
    if their interruption probability is above a threshold.
    """
    ON_DEMAND = "on-demand"
    SPOT = "spot"


class JobState(str, Enum):
    """
    Lifecycle states of a job execution.

    PENDING    → Admitted, waiting in the priority queue.
    SCHEDULED  → ACO assigned it to a node; resources reserved.
    RUNNING    → Job is actively executing on its node.
    COMPLETED  → Finished successfully; resources released.
    FAILED     → Terminated with error; resources released.
    PREEMPTED  → Evicted to make room for a higher-priority job.
                 Will be re-queued as PENDING.
    """
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: TELEMETRY MODELS
# What a node reports about itself at runtime.
# ─────────────────────────────────────────────────────────────────────────────

class NodeTelemetry(BaseModel):
    """
    A single heartbeat snapshot from a compute node.

    Why we need this (the V1 gap):
        V1 only tracked *allocated* resources — what the scheduler
        *intended* to give each node. It never tracked *actual* utilisation.

        A node could be at 100% CPU utilisation while showing 2 free cores
        in the allocator (because a job was using more than it requested).
        V2 closes this gap by ingesting real metrics from the node agent.

    Fields:
        node_id         → Which node this snapshot is for.
        timestamp       → When this reading was taken (UTC).
        cpu_util_pct    → Actual CPU utilisation 0–100.
                          Compare with: node.allocated_cpu_cores / node.total_cpu_cores * 100
                          If cpu_util_pct >> allocated%, the node is hot.
        memory_util_pct → Actual memory utilisation 0–100.
        gpu_util_pct    → Per GPU-model utilisation map.
                          e.g., {"A100": 87.5, "V100": 12.0}
                          Empty dict if no GPUs on this node.
        power_watts     → Total node power draw. Used for energy-aware scheduling
                          and cost estimation (power × time × $/kWh).
        network_rx_mbps → Network receive throughput. Useful for STREAM workloads
                          to detect bandwidth saturation.
        network_tx_mbps → Network transmit throughput.
        temperature_c   → CPU die temperature. If above threshold (~85°C),
                          the scheduler should deprioritise this node (thermal throttle
                          is already reducing effective performance).
    """
    node_id: str = Field(..., description="Which node this snapshot is from")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time of this reading"
    )

    # CPU & Memory — actual utilisation (not just allocated)
    cpu_util_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Actual CPU utilisation as a percentage (0-100)"
    )
    memory_util_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Actual memory utilisation as a percentage (0-100)"
    )

    # GPU — per-model actual utilisation
    gpu_util_pct: Dict[str, float] = Field(
        default_factory=dict,
        description="GPU utilisation per model, e.g. {'A100': 87.5}. Empty if no GPUs."
    )

    # Power and thermal
    power_watts: float = Field(
        0.0, ge=0.0,
        description="Total node power draw in watts"
    )
    temperature_c: Optional[float] = Field(
        None,
        description="CPU die temperature in Celsius. None if sensor unavailable."
    )

    # Network throughput (relevant for STREAM workloads)
    network_rx_mbps: float = Field(0.0, ge=0.0, description="Receive throughput in Mbps")
    network_tx_mbps: float = Field(0.0, ge=0.0, description="Transmit throughput in Mbps")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RESOURCE MODELS
# What a job asks for and what a node provides.
# ─────────────────────────────────────────────────────────────────────────────

class ResourceRequest(BaseModel):
    """
    The compute resources a job requires.

    V1 had: cpu_cores_min, memory_gb_min, gpu_required (bool), cpu_cores_max, memory_gb_max
    V2 adds: gpu_count, gpu_memory_gb, network_bandwidth_mbps

    Why gpu_count matters:
        V1 hardcoded `required_count = 1` in the scheduler.
        A large ML training job might need 8× A100s on a single node.
        Without gpu_count, the scheduler can't correctly gate placement.

    Why gpu_memory_gb matters:
        Two A100 variants exist: 40GB and 80GB VRAM.
        A model that doesn't fit in 40GB will OOM and fail.
        The scheduler must match VRAM requirement, not just GPU count.

    Why network_bandwidth_mbps matters:
        STREAM jobs reading from Kafka at 1 Gbps need a node with
        enough NIC headroom. Without this, placement succeeds but the
        job throughput-starves.
    """
    # CPU
    cpu_cores_min: float = Field(
        ..., gt=0,
        description="Minimum CPU cores required (fractional ok, e.g. 0.5)"
    )
    cpu_cores_max: Optional[float] = Field(
        None,
        description="Max CPU cores the job can burst to. Used for bin-packing overhead calc."
    )

    # Memory
    memory_gb_min: float = Field(
        ..., gt=0,
        description="Minimum RAM in GB required"
    )
    memory_gb_max: Optional[float] = Field(
        None,
        description="Max memory the job can burst to."
    )

    # GPU
    gpu_required: bool = Field(
        False,
        description="Does this job need a GPU at all?"
    )
    gpu_count: int = Field(
        1, ge=1,
        description="Number of GPUs required. Only relevant if gpu_required=True."
    )
    gpu_memory_gb: Optional[float] = Field(
        None, ge=0,
        description="Minimum VRAM required per GPU in GB. None = no constraint."
    )

    # Network
    network_bandwidth_mbps: Optional[float] = Field(
        None, ge=0,
        description="Minimum NIC bandwidth required. Relevant for STREAM workloads."
    )


class NodeCostProfile(BaseModel):
    """
    Billing and reliability characteristics of a node.

    Why this is a separate model:
        Cost and reliability are orthogonal to capacity.
        A node's cost_per_hour doesn't change when a job runs on it.
        Keeping it separate makes the cost engine's job cleaner.

    Fields:
        instance_type         → ON_DEMAND or SPOT (see InstanceType enum).
        cost_per_hour_usd     → Hourly rate for this node.
        interruption_prob     → Probability the node will be reclaimed in the
                                next hour (0.0 = never, 1.0 = certain).
                                For SPOT: derived from historical AWS/GCP spot data.
                                For ON_DEMAND: always 0.0.
        region                → AWS/GCP region string, e.g. "us-east-1".
                                Used to group nodes and estimate inter-node latency.
    """
    instance_type: InstanceType = Field(
        InstanceType.ON_DEMAND,
        description="Billing model: on-demand (stable) or spot (cheap but interruptible)"
    )
    cost_per_hour_usd: float = Field(
        0.0, ge=0.0,
        description="Hourly cost of this node in USD"
    )
    interruption_prob: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Probability of spot interruption in the next hour. 0.0 for on-demand."
    )
    region: str = Field(
        "us-east-1",
        description="Cloud region identifier"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: COMPUTE NODE
# Represents a physical or virtual machine in the cluster.
# ─────────────────────────────────────────────────────────────────────────────

class ComputeNode(BaseModel):
    """
    Full representation of a compute node in the cluster.

    V1 had: node_id, state, total_cpu_cores, total_memory_gb,
            gpu_inventory, allocated_cpu_cores, allocated_memory_gb, allocated_gpus

    V2 adds: arch, cost_profile, latest_telemetry

    Why arch matters for scheduling:
        An ARM64 Graviton node is ~20% cheaper per core than x86_64 for
        compute-bound workloads. But some jobs have x86-specific binaries.
        The scheduler must respect this constraint.

    Why cost_profile is on the node (not the job):
        Spot vs on-demand is a property of *where you run*, not *what you run*.
        The cost engine reads node.cost_profile to score placement candidates.

    Why latest_telemetry is Optional:
        At startup, nodes exist before any heartbeat arrives.
        The scheduler falls back to allocation-based estimates until
        real telemetry is available.
    """
    node_id: str = Field(..., description="Unique identifier for this node")
    state: NodeState = Field(NodeState.HEALTHY, description="Operational state")

    # Hardware topology
    arch: NodeArch = Field(
        NodeArch.X86_64,
        description="CPU architecture. Affects job compatibility and cost."
    )

    # Raw capacity
    total_cpu_cores: float = Field(..., ge=0, description="Total CPU cores on this node")
    total_memory_gb: float = Field(..., ge=0, description="Total RAM in GB")

    # GPU inventory: model_name → total count
    # e.g., {"A100": 8, "V100": 2}
    gpu_inventory: Dict[str, int] = Field(
        default_factory=dict,
        description="Map of GPU model to total count. Empty for CPU-only nodes."
    )

    # GPU VRAM per model: model_name → VRAM in GB per unit
    # e.g., {"A100": 80.0, "V100": 32.0}
    gpu_vram_gb: Dict[str, float] = Field(
        default_factory=dict,
        description="VRAM per GPU in GB, keyed by model. Used for VRAM-aware placement."
    )

    # Current allocations (what the scheduler has *promised* to jobs)
    allocated_cpu_cores: float = Field(0.0, ge=0, description="CPU cores reserved for running jobs")
    allocated_memory_gb: float = Field(0.0, ge=0, description="RAM reserved for running jobs")
    allocated_gpus: Dict[str, int] = Field(
        default_factory=dict,
        description="GPUs reserved per model. e.g., {'A100': 3}"
    )

    # Cost and billing (V2 addition)
    cost_profile: NodeCostProfile = Field(
        default_factory=NodeCostProfile,
        description="Billing model and spot interruption risk for this node"
    )

    # Live telemetry (V2 addition — populated by the telemetry collector)
    latest_telemetry: Optional[NodeTelemetry] = Field(
        None,
        description="Most recent heartbeat from this node's agent. None until first heartbeat."
    )

    # ── Derived properties (fast scheduling helpers) ──────────────────────────

    @property
    def available_cpu_cores(self) -> float:
        """CPU cores not yet promised to any job."""
        return self.total_cpu_cores - self.allocated_cpu_cores

    @property
    def available_memory_gb(self) -> float:
        """RAM not yet promised to any job."""
        return self.total_memory_gb - self.allocated_memory_gb

    @property
    def cpu_utilisation_pct(self) -> float:
        """
        Best estimate of actual CPU utilisation.

        Priority:
        1. Use real telemetry if we have it (actual OS-level measurement).
        2. Fall back to allocation ratio (what the scheduler *thinks* is used).

        Why this matters:
            Allocation ratio can be misleading. A job might be scheduled for
            4 cores but actually idle. Or it might be using 6 cores (overcommit).
            Real telemetry catches both cases.
        """
        if self.latest_telemetry:
            return self.latest_telemetry.cpu_util_pct
        # Fallback: allocation-based estimate
        if self.total_cpu_cores == 0:
            return 0.0
        return (self.allocated_cpu_cores / self.total_cpu_cores) * 100.0

    @property
    def memory_utilisation_pct(self) -> float:
        """Best estimate of actual memory utilisation. Same logic as cpu_utilisation_pct."""
        if self.latest_telemetry:
            return self.latest_telemetry.memory_util_pct
        if self.total_memory_gb == 0:
            return 0.0
        return (self.allocated_memory_gb / self.total_memory_gb) * 100.0

    @property
    def is_thermally_safe(self) -> bool:
        """
        True if the node is not thermal-throttling.

        Threshold: 85°C. Above this, modern Intel/AMD CPUs reduce clock speeds.
        If we don't have telemetry (or temperature sensor), we assume safe.
        """
        if self.latest_telemetry and self.latest_telemetry.temperature_c is not None:
            return self.latest_telemetry.temperature_c < 85.0
        return True  # Assume safe if unknown

    def available_gpus(self, model: str) -> int:
        """Free GPUs of a given model (inventory minus allocated)."""
        return self.gpu_inventory.get(model, 0) - self.allocated_gpus.get(model, 0)

    def can_fit(self, request: ResourceRequest) -> bool:
        """
        Quick feasibility check: can this node satisfy a resource request?

        Used by the bin-packer and ACO fast-path to filter candidate nodes
        before running the full scoring function.

        Checks (in order):
        1. Node is healthy (state check).
        2. CPU headroom.
        3. Memory headroom.
        4. GPU availability (count + VRAM).
        5. Thermal safety.
        """
        if self.state != NodeState.HEALTHY:
            return False
        if self.available_cpu_cores < request.cpu_cores_min:
            return False
        if self.available_memory_gb < request.memory_gb_min:
            return False
        if request.gpu_required:
            # Check each GPU type — we need at least one type that satisfies count + VRAM
            found_gpu = False
            for model, total in self.gpu_inventory.items():
                free = self.available_gpus(model)
                if free < request.gpu_count:
                    continue
                # VRAM check (only if job specifies a minimum)
                if request.gpu_memory_gb is not None:
                    vram = self.gpu_vram_gb.get(model, 0.0)
                    if vram < request.gpu_memory_gb:
                        continue
                found_gpu = True
                break
            if not found_gpu:
                return False
        if not self.is_thermally_safe:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: JOB MODELS
# What a user submits and how we track execution.
# ─────────────────────────────────────────────────────────────────────────────

class JobRequest(BaseModel):
    """
    The complete specification of a job submitted to the scheduler.

    V1 had: job_id, workload_type, resources, priority,
            latency_p99_ms, throughput_target_rps, gpu_model_preferred

    V2 adds: deadline_epoch, cost_ceiling_usd, preemptible, arch_required

    Why deadline_epoch matters:
        V1 had no concept of deadlines. A batch job could sit in the queue
        forever with no SLA violation detected. With a deadline, the scheduler
        can calculate urgency and boost priority as the deadline approaches.

    Why cost_ceiling_usd matters:
        If a user submits a job with cost_ceiling_usd=0.10, the scheduler
        must not place it on a $5/hr on-demand node. Cost-awareness prevents
        runaway spend.

    Why preemptible matters:
        A preemptible job explicitly agrees to be evicted if a higher-priority
        job arrives. This allows the scheduler to overcommit spot resources
        and achieve the 28% utilisation improvement target.

    Why arch_required matters:
        Some ML inference runtimes only ship x86_64 binaries.
        Some HPC jobs are compiled for ARM64 NEON intrinsics.
        If the job cares, it says so here.
    """
    job_id: str = Field(..., description="Unique job identifier (set by orchestrator)")
    workload_type: WorkloadType

    # Resource requirements
    resources: ResourceRequest

    # Scheduling priority: higher = scheduled first
    priority: int = Field(50, ge=1, le=100, description="Scheduling priority (1=lowest, 100=highest)")

    # ── SLA targets (workload-type specific) ─────────────────────────────────

    # For LATENCY_CRITICAL: the P99 latency the job must achieve
    latency_p99_ms: Optional[int] = Field(
        None, gt=0,
        description="P99 latency target in ms. Required for LATENCY_CRITICAL jobs."
    )

    # For BATCH / STREAM: throughput requirement
    throughput_target_rps: Optional[float] = Field(
        None, gt=0,
        description="Target processing rate (requests or items per second)."
    )

    # For GPU jobs: which GPU model is preferred
    gpu_model_preferred: Optional[str] = Field(
        None,
        description="Preferred GPU model e.g. 'A100', 'V100'. None = any GPU."
    )

    # ── V2 additions ──────────────────────────────────────────────────────────

    # Hard deadline (Unix epoch, seconds). None = no deadline.
    deadline_epoch: Optional[float] = Field(
        None,
        description="Unix timestamp by which this job must START (not finish). "
                    "Scheduler boosts priority as deadline approaches."
    )

    # Maximum this job is allowed to cost (per scheduling decision)
    cost_ceiling_usd: Optional[float] = Field(
        None, ge=0,
        description="Maximum hourly cost of the node this job can run on. "
                    "Prevents placement on expensive on-demand nodes if not needed."
    )

    # Can this job be evicted to make room for a higher-priority job?
    preemptible: bool = Field(
        False,
        description="If True, this job accepts eviction. Allows overcommit and "
                    "enables spot-instance placement for better utilisation."
    )

    # Architecture constraint
    arch_required: Optional[NodeArch] = Field(
        None,
        description="If set, the job will only run on nodes of this architecture. "
                    "None = architecture-agnostic."
    )


class JobExecution(BaseModel):
    """
    Tracks a live or completed job execution.

    V1 had: job_id, job_request, assigned_node_id, state,
            submitted_at, scheduled_at, started_at, completed_at,
            scheduling_latency_ms, failure_reason

    V2 adds: actual_cpu_used_cores, actual_memory_used_gb, preemption_count

    Why actual resource usage matters:
        V1 tracked what was *requested*. V2 tracks what was *used*.
        The difference feeds back into the WorkloadProfile (telemetry.py),
        which the predictor uses to forecast future demand.
        Over time, the scheduler learns that "BATCH jobs of type X usually
        need 6 cores even though they request 4."

    Why preemption_count matters:
        A job that gets preempted repeatedly is a signal that the cluster
        is under-provisioned for its priority class. Useful for capacity planning.
    """
    job_id: str
    job_request: JobRequest
    assigned_node_id: str
    state: JobState

    # Lifecycle timestamps
    submitted_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Performance metrics
    scheduling_latency_ms: float = Field(
        0.0,
        description="Time from submission to scheduled state (ms). Target: <10ms."
    )
    failure_reason: Optional[str] = None

    # V2: Actual resource consumption (populated by node agent on completion)
    actual_cpu_used_cores: Optional[float] = Field(
        None,
        description="Peak CPU cores observed during execution. Feeds WorkloadProfile."
    )
    actual_memory_used_gb: Optional[float] = Field(
        None,
        description="Peak memory observed during execution."
    )
    actual_gpu_util_pct: Optional[Dict[str, float]] = Field(
        None,
        description="Average GPU utilisation % during execution, per model."
    )

    # V2: How many times was this job evicted and re-queued?
    preemption_count: int = Field(
        0,
        description="Number of times this job was preempted. 0 = ran to completion uninterrupted."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: PREDICTION MODEL
# Output of the forecaster — consumed by the ACO cost engine.
# ─────────────────────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    """
    The output of the WorkloadPredictor for a single node over a forecast window.

    This is produced by orchestrator/control_plane/predictor.py (Phase 3)
    and consumed by:
    - The ACO cost engine (to penalise nodes predicted to be overloaded)
    - The telemetry collector (to trigger container pre-warming)
    - The /predict API endpoint (for observability)

    Fields:
        node_id               → Which node this prediction is for.
        forecast_horizon_min  → How far ahead we're predicting (e.g., 5 minutes).
        predicted_cpu_util    → Expected CPU utilisation % at end of horizon.
        predicted_memory_util → Expected memory utilisation % at end of horizon.
        predicted_gpu_util    → Expected GPU utilisation % per model.
        spike_probability     → Probability (0–1) of a utilisation spike
                                exceeding 90% CPU or GPU within the horizon.
                                > 0.7 → trigger container pre-warming.
        confidence            → How confident the model is (0–1).
                                Low confidence = more conservative placement.
        generated_at          → When this prediction was computed (for cache staleness).
    """
    node_id: str
    forecast_horizon_min: int = Field(5, description="Prediction window in minutes")

    predicted_cpu_util: float = Field(..., ge=0.0, le=100.0)
    predicted_memory_util: float = Field(..., ge=0.0, le=100.0)
    predicted_gpu_util: Dict[str, float] = Field(default_factory=dict)

    spike_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of >90% utilisation spike within forecast_horizon_min"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence. Low = prediction is uncertain."
    )
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: CONVENIENCE TYPE ALIASES
# Used throughout the codebase for readability.
# ─────────────────────────────────────────────────────────────────────────────

# A placement plan maps job_id → node_id
# e.g., {"job-abc123": "node-gpu-03", "job-def456": "node-cpu-01"}
PlacementPlan = Dict[str, str]

# Telemetry history: a list of snapshots ordered oldest-first
TelemetryHistory = List[NodeTelemetry]
