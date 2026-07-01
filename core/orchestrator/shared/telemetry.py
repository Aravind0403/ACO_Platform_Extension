"""
orchestrator/shared/telemetry.py
─────────────────────────────────
WorkloadProfile: a rolling statistical summary of how a job *type* behaves.

Why this is a separate file from models.py
------------------------------------------
models.py defines the *instantaneous* state of the world (a node right now,
a job right now). telemetry.py defines the *historical* knowledge the system
accumulates over time.

Think of it like this:
  models.py  → "What is happening right now?"
  telemetry.py → "What have we learned from the past?"

The predictor (Phase 3) reads WorkloadProfile to generate PredictionResult.
The orchestration service writes to WorkloadProfile every time a job completes.

How WorkloadProfile gets populated
------------------------------------
1. A job finishes. OrchestratorService calls complete_job().
2. complete_job() reads job_execution.actual_cpu_used_cores, actual_memory_used_gb, etc.
3. It updates the WorkloadProfile for that job's workload_type.
4. The predictor periodically reads all WorkloadProfiles to re-fit its model.

Over time, the scheduler learns:
  "BATCH jobs that request 4 cores actually use 6.2 on average."
  "STREAM jobs burst CPU by 40% every ~15 minutes."
  "LATENCY_CRITICAL jobs almost never use GPU even when they request it."
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class ResourceSample(BaseModel):
    """
    A single data point from one completed job execution.

    Stored in WorkloadProfile.samples and used to compute running statistics.
    We keep the raw samples (up to a cap) so the predictor can fit time-series
    models directly on them, not just on pre-computed averages.

    Fields:
        timestamp       → When this sample was recorded (job completion time).
        cpu_cores_used  → Peak CPU cores observed during execution.
        memory_gb_used  → Peak memory observed during execution.
        gpu_util_pct    → Average GPU utilisation % (0–100). None if no GPU.
        duration_s      → How long the job actually ran in seconds.
        scheduling_latency_ms → How long it took to schedule this job.
                                Used to track P99 scheduling latency over time.
    """
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_cores_used: float = Field(..., ge=0)
    memory_gb_used: float = Field(..., ge=0)
    gpu_util_pct: Optional[float] = Field(None, ge=0, le=100)
    duration_s: float = Field(..., ge=0)
    scheduling_latency_ms: float = Field(..., ge=0)


class WorkloadProfile(BaseModel):
    """
    A rolling statistical profile for a named workload class.

    What it does:
        Aggregates ResourceSamples from completed jobs to build a statistical
        picture of what this workload type typically consumes.

    What the predictor does with it:
        - Fits an exponential smoothing model on cpu_cores_history.
        - Looks at burst_factor to decide how much headroom to reserve.
        - Uses avg_duration_s to estimate when the node will free up.

    Fields:
        workload_name      → Human-readable name (matches WorkloadType value
                             or a user-defined sub-class like "ml-training-large").
        sample_count       → Total number of completed jobs contributing to this profile.
        samples            → Raw samples (capped at max_samples for memory).
                             Oldest samples are dropped when the cap is reached.
        max_samples        → Cap on stored samples. Default 500. Roughly 8 hours
                             at 1 job/minute.

        avg_cpu_cores      → Rolling mean of cpu_cores_used across all samples.
        avg_memory_gb      → Rolling mean of memory_gb_used.
        avg_gpu_util_pct   → Rolling mean of gpu_util_pct (None if no GPU samples).
        avg_duration_s     → Rolling mean of job duration.
        avg_scheduling_latency_ms → Rolling mean of scheduling latency. Used to
                                    verify we're hitting the <10ms target.

        burst_factor       → Ratio of max(cpu_cores_used) / avg(cpu_cores_used)
                             across recent samples. A burst_factor of 2.0 means
                             jobs sometimes spike to 2× their average.
                             The predictor uses this to add a safety margin.

        p99_latency_ms     → 99th percentile of observed scheduling latency.
                             Primary SLA metric. Target: <10ms.

        last_updated       → When this profile was last recalculated.
    """
    workload_name: str = Field(..., description="Workload type or sub-class name")
    sample_count: int = Field(0, ge=0, description="Total jobs contributing to this profile")
    max_samples: int = Field(500, description="Maximum raw samples to retain")
    samples: List[ResourceSample] = Field(
        default_factory=list,
        description="Raw samples. Capped at max_samples (FIFO drop)."
    )

    # Rolling statistics (recalculated each time a sample is added)
    avg_cpu_cores: float = Field(0.0, ge=0)
    avg_memory_gb: float = Field(0.0, ge=0)
    avg_gpu_util_pct: Optional[float] = Field(None)
    avg_duration_s: float = Field(0.0, ge=0)
    avg_scheduling_latency_ms: float = Field(0.0, ge=0)

    # Burst behaviour
    burst_factor: float = Field(
        1.0, ge=1.0,
        description="Max/avg CPU ratio. 1.0 = flat load. 2.0 = sometimes doubles."
    )

    # SLA tracking
    p99_latency_ms: float = Field(
        0.0, ge=0,
        description="P99 scheduling latency across all samples. Target: <10ms."
    )

    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def add_sample(self, sample: ResourceSample) -> None:
        """
        Add a new ResourceSample and recompute all statistics.

        Called by OrchestratorService.complete_job() after every job finishes.

        Algorithm:
        1. Append sample to self.samples (drop oldest if over cap).
        2. Recalculate all rolling statistics from the current window.

        We recalculate from the raw samples (not incrementally) because:
        - Simplicity: no incremental variance formula needed.
        - Correctness: when old samples are dropped, the running average
          automatically adjusts to the current window.
        - Performance: capped at 500 samples → negligible cost (<1ms).
        """
        # Append, then trim to cap (FIFO — drop from front)
        self.samples.append(sample)
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples:]

        self.sample_count += 1
        n = len(self.samples)

        # Recalculate from current window
        cpu_values = [s.cpu_cores_used for s in self.samples]
        mem_values = [s.memory_gb_used for s in self.samples]
        gpu_values = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        dur_values = [s.duration_s for s in self.samples]
        lat_values = [s.scheduling_latency_ms for s in self.samples]

        self.avg_cpu_cores = sum(cpu_values) / n
        self.avg_memory_gb = sum(mem_values) / n
        self.avg_gpu_util_pct = sum(gpu_values) / len(gpu_values) if gpu_values else None
        self.avg_duration_s = sum(dur_values) / n
        self.avg_scheduling_latency_ms = sum(lat_values) / n

        # Burst factor: max observed / mean (floor at 1.0 to avoid division edge cases)
        if self.avg_cpu_cores > 0:
            self.burst_factor = max(cpu_values) / self.avg_cpu_cores
        else:
            self.burst_factor = 1.0

        # P99 latency: sort latencies, take the 99th percentile value
        sorted_lat = sorted(lat_values)
        p99_idx = max(0, int(0.99 * n) - 1)
        self.p99_latency_ms = sorted_lat[p99_idx]

        self.last_updated = lambda: datetime.now(timezone.utc)()

    @property
    def cpu_cores_history(self) -> List[float]:
        """
        Ordered list of cpu_cores_used values (oldest first).

        Used by the predictor to fit a time-series model.
        Returns an empty list if no samples yet.
        """
        return [s.cpu_cores_used for s in self.samples]

    @property
    def scheduling_latency_history(self) -> List[float]:
        """
        Ordered list of scheduling_latency_ms values (oldest first).

        Used to plot the latency trend over time and verify
        the <10ms SLA is being maintained.
        """
        return [s.scheduling_latency_ms for s in self.samples]

    @property
    def has_enough_data(self) -> bool:
        """
        Returns True if we have enough samples to make a meaningful prediction.

        Threshold: 10 samples minimum.
        Below this, the predictor falls back to the raw average with no forecast.

        Why 10?
            Exponential smoothing needs at least a few data points to converge.
            10 is conservative but safe for a V2 implementation.
        """
        return len(self.samples) >= 10
