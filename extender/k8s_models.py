"""
extender/k8s_models.py
───────────────────────
Pydantic models for the Kubernetes Scheduler Extender HTTP protocol.

K8s sends these payloads to /filter and /prioritize.
We receive them, run ACO, and return the response shapes below.

Reference:
  https://kubernetes.io/docs/concepts/scheduling-eviction/scheduler-extensions/
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel


# ── What K8s sends us ────────────────────────────────────────────────────────

class ResourceList(BaseModel):
    """CPU and memory from a pod's resource requests."""
    cpu: Optional[str] = None       # e.g. "500m" or "2"
    memory: Optional[str] = None    # e.g. "256Mi" or "4Gi"


class NodeCapacity(BaseModel):
    """Allocatable resources on a node (from K8s NodeInfo)."""
    cpu: Optional[str] = None
    memory: Optional[str] = None


class NodeInfo(BaseModel):
    """Minimal node representation K8s passes to extender."""
    name: str
    capacity: Optional[NodeCapacity] = None
    allocatable: Optional[NodeCapacity] = None
    labels: Optional[Dict[str, str]] = None


class NodeList(BaseModel):
    items: List[NodeInfo] = []


class PodSpec(BaseModel):
    """Relevant slice of a pod spec — resource requests + GPU flag."""
    containers: Optional[List[Dict]] = None
    nodeName: Optional[str] = None


class Pod(BaseModel):
    metadata: Optional[Dict] = None
    spec: Optional[PodSpec] = None


class ExtenderArgs(BaseModel):
    """
    The payload K8s POSTs to /filter and /prioritize.

    Pod   — the pod being scheduled
    Nodes — list of candidate nodes (after default scheduler pre-filter)
    """
    Pod: Pod
    Nodes: Optional[NodeList] = None
    NodeNames: Optional[List[str]] = None


# ── What we send back ────────────────────────────────────────────────────────

class ExtenderFilterResult(BaseModel):
    """
    Response to /filter.
    Nodes  — the subset of input nodes we deem viable
    FailedNodes — dict of node_name → reason for nodes we rejected
    Error  — non-empty string causes K8s to skip the extender this cycle
    """
    Nodes: NodeList = NodeList()
    NodeNames: Optional[List[str]] = None
    FailedNodes: Dict[str, str] = {}
    Error: str = ""


class HostPriority(BaseModel):
    """Score for one node. K8s picks the highest-scoring node overall."""
    Host: str
    Score: int   # 0–10


from pydantic import RootModel

class HostPriorityList(RootModel[List[HostPriority]]):
    """Response to /prioritize — list of (node, score) pairs."""

    def __iter__(self):
        return iter(self.root)
