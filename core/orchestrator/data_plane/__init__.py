"""
orchestrator/data_plane — per-node execution agents.

Phase 8: NodeAgent simulates job execution and sends telemetry heartbeats.

Public API:
    NodeAgent   — async per-node daemon (execute jobs, send heartbeats)
"""

from orchestrator.data_plane.agent import NodeAgent

__all__ = ["NodeAgent"]
