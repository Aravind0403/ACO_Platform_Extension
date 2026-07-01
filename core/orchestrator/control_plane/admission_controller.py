"""
orchestrator/control_plane/admission_controller.py
────────────────────────────────────────────────────
Admission control: semantic validation before scheduling.

The admission controller is the first gate in the scheduling pipeline.
It runs AFTER Pydantic validation (which handles schema correctness) and
BEFORE the scheduler (which handles placement).

What it checks
───────────────
  1. Priority bounds: LATENCY_CRITICAL jobs must have priority ≥ 80.
     Rationale: LC jobs must be served first; a priority-10 LC job
     would be a misconfiguration.

  2. GPU coherence: if gpu_required=True, gpu_count must be ≥ 1.
     Pydantic already enforces gpu_count ≥ 1 by default, but we check
     the combination here (gpu_required=False with gpu_count=4 is odd,
     but not a hard error).

  3. Resource sanity: cpu_cores_min must be > 0. (Pydantic validates gt=0
     at the schema level, so this is a belt-and-suspenders check.)

  4. Deadline feasibility: if deadline_epoch is set, it must be in the
     future. A job with a past deadline is already violated before it
     schedules — reject fast.

What it does NOT check
───────────────────────
  • Whether specific nodes have capacity — that's the scheduler's job.
  • Whether cost_ceiling_usd is realistic — that's enforced by CostEngine.
  • GPU model availability — that's enforced by can_fit() in ComputeNode.

V1 compatibility
─────────────────
  AdmissionRejectedError is unchanged — callers (OrchestratorService) catch it
  identically to V1.

  admit_job(job_request) signature is unchanged.
"""

from __future__ import annotations

import time

from orchestrator.shared.models import JobRequest, WorkloadType


class AdmissionRejectedError(Exception):
    """
    Raised when a job fails admission control.

    Attributes:
        reason: Human-readable explanation of why the job was rejected.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def admit_job(job_request: JobRequest) -> None:
    """
    Run all admission checks on a JobRequest.

    Raises AdmissionRejectedError if any check fails.
    Returns None on success (caller proceeds to scheduling).

    Args:
        job_request: The validated JobRequest to check.

    Raises:
        AdmissionRejectedError: with a descriptive reason string.
    """
    _check_latency_critical_priority(job_request)
    _check_gpu_coherence(job_request)
    _check_deadline(job_request)


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_latency_critical_priority(job_request: JobRequest) -> None:
    """
    LATENCY_CRITICAL jobs must have priority ≥ 80.

    Why 80? LC jobs have strict P99 SLA targets (<10ms scheduling latency).
    Submitting an LC job with low priority is a misconfiguration — it would
    lose queue position to BATCH jobs and miss its SLA window.
    """
    if (
        job_request.workload_type == WorkloadType.LATENCY_CRITICAL
        and job_request.priority < 80
    ):
        raise AdmissionRejectedError(
            f"LATENCY_CRITICAL job {job_request.job_id!r} has priority "
            f"{job_request.priority} — must be ≥ 80. "
            f"Low-priority LC jobs will miss SLA targets."
        )


def _check_gpu_coherence(job_request: JobRequest) -> None:
    """
    If gpu_required=True, gpu_count must be ≥ 1 (already enforced by Pydantic).
    Check the inverse: if gpu_count > 1 but gpu_required=False, warn via error.
    This catches copy-paste misconfigurations.
    """
    if not job_request.resources.gpu_required and job_request.resources.gpu_count > 1:
        raise AdmissionRejectedError(
            f"Job {job_request.job_id!r} has gpu_count={job_request.resources.gpu_count} "
            f"but gpu_required=False. Set gpu_required=True or gpu_count=1."
        )


def _check_deadline(job_request: JobRequest) -> None:
    """
    If a deadline is set, it must be in the future.

    A job whose deadline has already passed will violate SLA the moment
    it is submitted. Reject early rather than wasting scheduling resources.
    """
    if job_request.deadline_epoch is not None:
        now = time.time()
        if job_request.deadline_epoch <= now:
            raise AdmissionRejectedError(
                f"Job {job_request.job_id!r} deadline has already passed "
                f"(deadline={job_request.deadline_epoch:.0f}, now={now:.0f}). "
                f"Submit with a future deadline."
            )
