"""
orchestrator/data_plane/agent.py
─────────────────────────────────
NodeAgent — lightweight per-node async daemon.

Role in the V2 architecture
─────────────────────────────
After aco_schedule() places a job on a node, someone must actually execute it,
measure resource consumption, and close the feedback loop by calling
complete_job() with real CPU/memory figures. That is NodeAgent's job.

V2 design (in-process)
────────────────────────
NodeAgent runs inside the same Python process as OrchestratorService. Direct
method calls replace HTTP — no network stack, no serialisation overhead.
The asyncio event loop keeps execution and heartbeats concurrent without threads.

V3 upgrade path
────────────────
Replace every ``self._service.X(...)`` call with ``await self._http.post("/X")``.
The rest of the agent (execution model, heartbeat loop, resource sampling) stays
identical.

Execution model
───────────────
Job duration is proportional to the fraction of the node's CPU requested:

    cpu_fraction  = requested_cpu_cores / node.total_cpu_cores
    base_duration = cpu_fraction × DURATION_SCALE_S
    duration      = clamp(gauss(base_duration, base_duration × 0.20), MIN, MAX)

Actual resource consumption is sampled around realistic ratios:

    actual_cpu = clamp(gauss(requested × CPU_USAGE_RATIO, noise), 0, requested × 1.1)
    actual_mem = clamp(gauss(requested × MEM_USAGE_RATIO, noise), 0, requested × 1.1)

The 10% over-cap (× 1.1) mimics how jobs occasionally burst beyond their
reservation, matching real Kubernetes resource limit behaviour.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Dict, Optional

from orchestrator.shared.models import JobExecution, NodeArch, NodeTelemetry

if TYPE_CHECKING:
    from orchestrator.control_plane.orchestration_service import OrchestratorService
    from orchestrator.telemetry.trace_adapter import AlibabaMachineTraceAdapter

logger = logging.getLogger(__name__)

# ── Execution model constants ────────────────────────────────────────────────────

# Duration
DURATION_SCALE_S: float = 10.0    # base multiplier: fraction_of_node_cpu × this
DURATION_NOISE_RATIO: float = 0.20  # Gaussian std as fraction of base (±20%)
MIN_DURATION_S: float = 0.05      # floor prevents zero-sleep in fast tests
MAX_DURATION_S: float = 30.0      # ceiling keeps tests tractable

# Actual resource usage vs. requested
CPU_USAGE_RATIO: float = 0.85     # jobs use ~85% of their reserved CPU on average
MEM_USAGE_RATIO: float = 0.90     # jobs use ~90% of their reserved memory on average
USAGE_NOISE_STD: float = 0.05     # Gaussian std (as fraction of requested)
USAGE_BURST_CAP: float = 1.10     # jobs may burst at most 10% above reservation

# Heartbeat
HEARTBEAT_INTERVAL_S: float = 5.0   # push NodeTelemetry every 5 seconds
HEARTBEAT_CPU_NOISE: float = 5.0    # Gaussian noise added to synthetic CPU reading

# GPU arch identifier (mirrors orchestrator/shared/models.py)
_GPU_ARCH = NodeArch.GPU_NODE


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class NodeAgent:
    """
    Simulated per-node execution daemon.

    One NodeAgent instance per ComputeNode. Responsibilities:
      1. ``execute_job()`` — async simulation of a job lifecycle.
      2. ``start()`` / ``stop()`` — background heartbeat task management.
      3. ``_build_telemetry()`` — synthesises NodeTelemetry for the heartbeat.

    Usage (synchronous test context):
        agent = NodeAgent("node-cpu-01", svc)
        await agent.execute_job(job_execution)
        # job is now in svc.completed_jobs with real CPU/memory figures

    Usage (long-running background context):
        agent = NodeAgent("node-gpu-04", svc)
        await agent.start()           # heartbeat begins
        await agent.execute_job(ex)   # concurrent with heartbeat
        await agent.stop()            # cancel heartbeat gracefully
    """

    def __init__(
        self,
        node_id: str,
        orchestration_service: "OrchestratorService",
        trace_adapter: Optional["AlibabaMachineTraceAdapter"] = None,
    ) -> None:
        """
        Create a NodeAgent bound to a specific node.

        Args:
            node_id:                 Must exist in orchestration_service.node_state.
            orchestration_service:   The live OrchestratorService.
            trace_adapter:           Optional Alibaba trace adapter. When provided,
                                     heartbeat telemetry uses real trace data instead
                                     of a synthetic utilisation estimate.

        Raises:
            ValueError: If node_id is not registered in the service.
        """
        if node_id not in orchestration_service.node_state:
            raise ValueError(
                f"NodeAgent: node_id {node_id!r} not found in OrchestratorService. "
                f"Known nodes: {list(orchestration_service.node_state)}"
            )
        self._node_id = node_id
        self._service = orchestration_service
        self._trace_adapter = trace_adapter

        # In-flight job tracking: job_id → JobExecution
        self._running_jobs: Dict[str, JobExecution] = {}

        # Background heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_tick: int = 0
        self._stopped: bool = False

    # ── Public API ───────────────────────────────────────────────────────────────

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def running_jobs(self) -> Dict[str, JobExecution]:
        """Read-only view of currently executing jobs."""
        return dict(self._running_jobs)

    async def execute_job(self, job_execution: JobExecution) -> None:
        """
        Simulate execution of one job and report completion to OrchestratorService.

        The method:
          1. Registers the job as in-flight.
          2. Sleeps for a duration proportional to the job's CPU request (cooperative).
          3. Samples actual CPU/memory consumption (Gaussian around realistic ratios).
          4. Calls service.complete_job() with the real figures.
          5. Removes the job from the in-flight registry.

        Args:
            job_execution: A JobExecution record for a job that is currently
                           in ``service.active_jobs`` (was created by submit_job()).
        """
        job_id = job_execution.job_id
        self._running_jobs[job_id] = job_execution

        try:
            duration = self._compute_duration(job_execution)
            logger.debug(
                "NodeAgent[%s]: executing job %s (simulated %.2fs)",
                self._node_id, job_id, duration,
            )
            await asyncio.sleep(duration)

            actual_cpu, actual_mem = self._sample_usage(job_execution)

            self._service.complete_job(
                job_id=job_id,
                success=True,
                actual_cpu_used_cores=actual_cpu,
                actual_memory_used_gb=actual_mem,
            )
            logger.debug(
                "NodeAgent[%s]: job %s complete (cpu=%.2f mem=%.2fGB)",
                self._node_id, job_id, actual_cpu, actual_mem,
            )
        except asyncio.CancelledError:
            # Agent was stopped mid-execution — report failure so resources are freed
            self._service.complete_job(
                job_id=job_id,
                success=False,
                failure_reason="NodeAgent cancelled during execution",
            )
            raise
        finally:
            self._running_jobs.pop(job_id, None)

    async def start(self) -> None:
        """
        Start the background heartbeat loop.

        Idempotent — safe to call multiple times; starts only one task.
        """
        if self._stopped:
            return
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(),
                name=f"heartbeat:{self._node_id}",
            )
            logger.debug("NodeAgent[%s]: heartbeat started", self._node_id)

    async def stop(self) -> None:
        """
        Gracefully cancel the heartbeat task.

        Safe to call before start() — no-op in that case.
        """
        self._stopped = True
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            logger.debug("NodeAgent[%s]: heartbeat stopped", self._node_id)

    # ── Internal helpers ─────────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Send NodeTelemetry to OrchestratorService every HEARTBEAT_INTERVAL_S."""
        try:
            while not self._stopped:
                telemetry = self._build_telemetry()
                self._service.update_node_telemetry(telemetry)
                self._heartbeat_tick += 1
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
        except asyncio.CancelledError:
            pass  # clean shutdown

    def _build_telemetry(self) -> NodeTelemetry:
        """
        Build a NodeTelemetry snapshot for this heartbeat tick.

        Trace-replay path (trace_adapter provided):
            Uses real Alibaba CPU/memory values for this node (per-node offset + scaling).

        Synthetic path (no adapter):
            Estimates CPU from in-flight job load:
              cpu_util = (sum of in-flight requested cores / total cores) × 100 + noise
            Memory: Gaussian around 50% baseline.
        """
        node = self._service.node_state[self._node_id]

        if self._trace_adapter is not None:
            cpu_util, mem_util = self._trace_adapter.get_reading(
                self._node_id, self._heartbeat_tick
            )
        else:
            # Synthetic: load from in-flight jobs + Gaussian noise
            running_cpu = sum(
                j.job_request.resources.cpu_cores_min
                for j in self._running_jobs.values()
            )
            raw_cpu = (running_cpu / node.total_cpu_cores) * 100.0
            cpu_util = _clamp(
                random.gauss(raw_cpu, HEARTBEAT_CPU_NOISE), 0.0, 100.0
            )
            mem_util = _clamp(random.gauss(50.0, 10.0), 0.0, 100.0)

        # GPU nodes get synthetic GPU utilisation (Alibaba trace has no GPU signal)
        gpu_util: Dict[str, float] = {}
        if node.arch == _GPU_ARCH:
            gpu_util = {
                "A100": _clamp(random.gauss(70.0, 15.0), 0.0, 100.0)
            }

        return NodeTelemetry(
            node_id=self._node_id,
            cpu_util_pct=cpu_util,
            memory_util_pct=mem_util,
            gpu_util_pct=gpu_util,
        )

    def _compute_duration(self, job_execution: JobExecution) -> float:
        """
        Compute a simulated job duration proportional to CPU request.

        Formula:
            cpu_fraction  = requested_cpu / total_cpu
            base_duration = cpu_fraction × DURATION_SCALE_S
            duration      = clamp(gauss(base, base × NOISE_RATIO), MIN, MAX)
        """
        node = self._service.node_state[self._node_id]
        cpu_fraction = (
            job_execution.job_request.resources.cpu_cores_min / node.total_cpu_cores
        )
        base = cpu_fraction * DURATION_SCALE_S
        noise = base * DURATION_NOISE_RATIO
        return _clamp(random.gauss(base, max(noise, 0.001)), MIN_DURATION_S, MAX_DURATION_S)

    def _sample_usage(self, job_execution: JobExecution) -> tuple[float, float]:
        """
        Sample actual CPU and memory consumption for a completed job.

        Actual values are drawn from a Gaussian centred at:
          actual_cpu = requested_cpu × CPU_USAGE_RATIO (0.85)
          actual_mem = requested_mem × MEM_USAGE_RATIO (0.90)

        Both clamped to (0, requested × USAGE_BURST_CAP] so the agent
        can never report more than 10% above the job's reservation.

        Returns:
            (actual_cpu_cores, actual_memory_gb)
        """
        req = job_execution.job_request.resources
        noise = USAGE_NOISE_STD

        actual_cpu = _clamp(
            random.gauss(req.cpu_cores_min * CPU_USAGE_RATIO, req.cpu_cores_min * noise),
            0.0,
            req.cpu_cores_min * USAGE_BURST_CAP,
        )
        actual_mem = _clamp(
            random.gauss(req.memory_gb_min * MEM_USAGE_RATIO, req.memory_gb_min * noise),
            0.0,
            req.memory_gb_min * USAGE_BURST_CAP,
        )
        return actual_cpu, actual_mem

    def __repr__(self) -> str:
        return (
            f"NodeAgent(node_id={self._node_id!r}, "
            f"running={len(self._running_jobs)}, "
            f"heartbeat={'running' if self._heartbeat_task and not self._heartbeat_task.done() else 'stopped'})"
        )
