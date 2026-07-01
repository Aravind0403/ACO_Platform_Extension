"""
aco_core — Ant Colony Optimisation scheduler core.

Public API:
    Colony           — run the ACO colony, returns PlacementPlan
    ColonyFailedError — raised when no feasible placement exists

Usage:
    from aco_core import Colony, ColonyFailedError

    colony = Colony(jobs=job_list, nodes=node_list)
    try:
        plan = colony.run()          # Dict[job_id, node_id]
    except ColonyFailedError:
        plan = naive_fallback(...)   # caller's responsibility
"""

from aco_core.colony import Colony, ColonyFailedError

__all__ = ["Colony", "ColonyFailedError"]
