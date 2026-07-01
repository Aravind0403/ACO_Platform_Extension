"""
scripts/trace_replay.py
────────────────────────
Local demo driver — fires scheduling requests at the ACO extender
to simulate what K8s would send during real cluster operation.

This lets you run a full demo without GKE or minikube:
  1. Start the extender:   uvicorn extender.main:app --port 8080
  2. Start observability:  cd observability && docker-compose up -d
  3. Run this script:      python scripts/trace_replay.py
  4. Open Grafana:         http://localhost:3000  (admin/admin)

Watch the pheromone panel converge as ACO learns to prefer
cheap, stable nodes over expensive ones.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from typing import Dict, List

import urllib.request
import urllib.error

# ── Simulated cluster topology ────────────────────────────────────────────────
# These match the GKE node pool labels in terraform/main.tf.
# In local mode these are just dicts — no real K8s needed.

NODES = [
    # CPU nodes
    {"name": "cpu-node-0", "allocatable": {"cpu": "4", "memory": "15Gi"},
     "labels": {"aco/gpu-type": "none", "aco/instance-type": "on_demand", "aco/cost-per-hour": "0.19", "aco/arch": "x86_64"}},
    {"name": "cpu-node-1", "allocatable": {"cpu": "4", "memory": "15Gi"},
     "labels": {"aco/gpu-type": "none", "aco/instance-type": "on_demand", "aco/cost-per-hour": "0.19", "aco/arch": "x86_64"}},
    # GPU nodes — one per tier
    {"name": "t4-node-0", "allocatable": {"cpu": "4", "memory": "15Gi"},
     "labels": {"aco/gpu-type": "t4", "aco/gpu-count": "1", "aco/instance-type": "on_demand", "aco/cost-per-hour": "0.45"}},
    {"name": "a10-node-0", "allocatable": {"cpu": "12", "memory": "85Gi"},
     "labels": {"aco/gpu-type": "a10", "aco/gpu-count": "1", "aco/instance-type": "on_demand", "aco/cost-per-hour": "1.20"}},
    {"name": "p100-node-0", "allocatable": {"cpu": "8", "memory": "30Gi"},
     "labels": {"aco/gpu-type": "p100", "aco/gpu-count": "1", "aco/instance-type": "on_demand", "aco/cost-per-hour": "1.60"}},
    {"name": "v100-node-0", "allocatable": {"cpu": "8", "memory": "30Gi"},
     "labels": {"aco/gpu-type": "v100", "aco/gpu-count": "1", "aco/instance-type": "on_demand", "aco/cost-per-hour": "2.48"}},
]

# ── Workload mix ──────────────────────────────────────────────────────────────
# (workload_type, cpu_req, mem_req_gi, gpu_required, priority, tenant)

WORKLOADS = [
    ("batch",            "2",    "4Gi",   False, 30,  "research"),
    ("batch",            "4",    "8Gi",   False, 50,  "prod"),
    ("batch",            "1",    "2Gi",   False, 20,  "dev"),
    ("latency-critical", "2",    "4Gi",   False, 90,  "prod"),
    ("latency-critical", "1",    "2Gi",   False, 95,  "prod"),
    ("stream-processing","2",    "8Gi",   False, 60,  "research"),
    ("batch",            "4",    "16Gi",  True,  40,  "research"),   # GPU batch
    ("batch",            "4",    "16Gi",  True,  80,  "prod"),       # GPU high-pri
    ("batch",            "2",    "8Gi",   True,  30,  "dev"),        # GPU dev
]


def _build_extender_args(workload: tuple, nodes: List[Dict]) -> Dict:
    wt, cpu, mem, gpu, priority, tenant = workload
    containers = [{
        "name": "job",
        "resources": {
            "requests": {"cpu": cpu, "memory": mem},
            "limits": ({"nvidia.com/gpu": "1"} if gpu else {}),
        },
    }]
    # Filter nodes to GPU-capable ones if job needs GPU
    candidate_nodes = [
        n for n in nodes
        if not gpu or n["labels"].get("aco/gpu-type", "none") != "none"
    ] if gpu else nodes

    return {
        "Pod": {
            "metadata": {
                "name": f"job-{uuid.uuid4().hex[:8]}",
                "labels": {
                    "aco/workload-type": wt,
                    "aco/tenant": tenant,
                },
            },
            "spec": {"containers": containers},
        },
        "Nodes": {"items": candidate_nodes},
    }


def _post(url: str, payload: Dict) -> Dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def run(extender_url: str, interval: float, count: int, verbose: bool):
    print(f"Starting trace replay → {extender_url}")
    print(f"  {len(NODES)} nodes | {len(WORKLOADS)} workload types | interval={interval}s")
    if count > 0:
        print(f"  Will fire {count} jobs then stop.")
    else:
        print("  Running indefinitely. Ctrl+C to stop.")
    print()

    sent = 0
    try:
        while count == 0 or sent < count:
            workload = random.choice(WORKLOADS)
            args = _build_extender_args(workload, NODES)
            wt, cpu, mem, gpu, priority, tenant = workload

            try:
                # Step 1: filter
                filter_result = _post(f"{extender_url}/filter", args)
                viable_nodes = filter_result.get("Nodes", {}).get("items", [])

                if not viable_nodes:
                    print(f"  [{sent+1}] {tenant}/{wt} → all nodes filtered out (skip)")
                    sent += 1
                    time.sleep(interval)
                    continue

                # Step 2: prioritize (only viable nodes)
                args["Nodes"]["items"] = viable_nodes
                priority_result = _post(f"{extender_url}/prioritize", args)

                if priority_result:
                    best = max(priority_result, key=lambda x: x["Score"])
                    if verbose:
                        scores = ", ".join(f"{r['Host']}={r['Score']}" for r in sorted(priority_result, key=lambda x: -x["Score"]))
                        print(f"  [{sent+1}] tenant={tenant} type={wt} gpu={gpu} → {best['Host']} (score={best['Score']}) | {scores}")
                    else:
                        print(f"  [{sent+1}] tenant={tenant:<10} {wt:<20} gpu={str(gpu):<5} → {best['Host']}")

            except urllib.error.URLError as e:
                print(f"  ✗ Connection failed: {e}. Is the extender running on {extender_url}?")
                time.sleep(2)
                continue

            sent += 1
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nStopped after {sent} jobs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACO trace replay — local demo driver")
    parser.add_argument("--url",      default="http://localhost:8080", help="Extender base URL")
    parser.add_argument("--interval", type=float, default=1.0,         help="Seconds between jobs")
    parser.add_argument("--count",    type=int,   default=0,           help="Jobs to fire (0=infinite)")
    parser.add_argument("--verbose",  action="store_true",             help="Print per-node scores")
    args = parser.parse_args()
    run(args.url, args.interval, args.count, args.verbose)
