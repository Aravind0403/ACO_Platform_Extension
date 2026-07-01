# Substack Part 2 Draft
## "The Algorithm Was Right. The Infrastructure Was Missing."

*Continuation of: [Your Scheduler Is Lying to You (And Ants Fixed It)](https://aravindsundaresan.substack.com/p/your-scheduler-is-lying-to-you-and)*

---

Start here if you haven't read Part 1: I built a compute scheduler that uses Ant Colony Optimisation to place GPU and CPU workloads on heterogeneous clusters. The colony accumulates pheromone trails across scheduling calls — nodes that receive good placements get reinforced, nodes that perform badly evaporate. A `CostEngine` heuristic multiplies the pheromone score by four factors: reliability, cost efficiency, SLA headroom, and spike prediction.

To find the right spike predictor, the paper ran a 14-way ablation across LSTM, GRU, TCN, Transformer, ARIMA, EMA, moving averages, and persistence models — 10 seeds, 200 latency-critical jobs, 32-node cluster. The result was unexpected: EMA with α=0.5 beat LSTM by 13.8 percentage points on safe-node routing (89.2% vs 75.4%). No training cost. No cold-start. One line of arithmetic. That result was submitted to a peer-reviewed systems conference (currently under review).

[Part 1](https://aravindsundaresan.substack.com/p/your-scheduler-is-lying-to-you-and) told that story — why first-fit scheduling lies to you, how ant colonies solve the problem, what the 7-benchmark honest scorecard showed including the failure (spike recall: 0%, because the Alibaba 2018 CPU trace didn't have enough spikes to train on). 0.82ms P99. 316x pheromone convergence magnitude. 96% cheaper than naive placement.

What Part 1 didn't say: all of that ran in Python dicts.

No Kubernetes. No Prometheus. No GPU node pools. The "cluster" was a dictionary keyed by node name. The "scheduling call" was a function call. The algorithm was correct. The infrastructure was a stub.

Building the platform layer is what happened next, and it broke in ways the simulation never revealed.

---

## The Gap Between Simulation and Deployment

Simulation is a controlled environment by definition. You set the nodes, the workloads, the trace. You control what "a scheduling call" means. The algorithm has no surface area for surprises.

Kubernetes is not a controlled environment. It has opinions. The scheduler has its own pipeline — predicate filters, priority functions, binding. If you want to change where pods land, you have to speak its language, not replace it. And its language, at the extension layer, is HTTP.

The Kubernetes scheduler extender protocol is how you wire a custom scorer into an existing cluster without forking anything. After the default scheduler runs its own pre-filters, it posts the surviving candidate list to your service's `/filter` endpoint. Your service narrows the list based on hard constraints. The survivors go to `/prioritize`, where your service scores each node 0–10. Kubernetes picks the winner.

Two HTTP endpoints. That's the entire integration surface.

What nobody tells you: the gap between "two HTTP endpoints" and "actually works under a scheduling call" is where most of the build time went.

---

## What Broke First

**PyTorch.** The research repo's `__init__.py` eagerly imports `WorkloadPredictor`, which pulls in PyTorch. PyTorch takes 8 seconds to load and 2GB of memory. A K8s sidecar that needs to respond to scheduling calls in <1ms cannot spend 8 seconds on startup.

The fix wasn't to restructure the research repo — that would couple the production extender to the research codebase, which will keep changing. The fix was `importlib.util.spec_from_file_location`: load `cost_engine.py` and the model files directly from their paths, bypassing the package `__init__` entirely. PyTorch never loads. Cold start drops to under 200ms.

**Pydantic v2.** The extender returns a list of `(node, score)` pairs to Kubernetes. The original model used Pydantic v1's `__root__` pattern. Pydantic v2 dropped it — not at model definition time, which would have been easy to catch, but at response serialisation time, which only surfaces during an actual scheduling call. A `TypeError` with no clear stack trace pointing to the cause. Two hours tracing through FastAPI's response pipeline to find a two-line fix.

**The gpu_count validator.** `CostEngine` enforces `gpu_count >= 1` as a Pydantic field validator. For CPU-only workloads, the original code passed `gpu_count=0`. This raised a `ValidationError` on every non-GPU scheduling call — silently, because the extender was catching the exception and returning an empty node list, which Kubernetes interpreted as "no viable nodes". Pods were being queued indefinitely. The fix: always pass `gpu_count=1`. The value is irrelevant for non-GPU jobs; the validator is satisfied either way.

None of these failures appeared in simulation. All three appeared within the first 20 minutes of running the extender against a live trace replay.

---

## Per-Tenant Pheromone

Part 1 had a single global pheromone table. One dict, all jobs sharing the same colony memory.

That's fine in simulation when you control all the workloads. In a real multi-tenant cluster, it's a contamination problem. A `prod` burst that drives heavy placement on the V100 tier leaves strong pheromone trails there. When the `research` tenant submits a job five seconds later, the colony inherits those trails — not because V100 is the right node for the research workload, but because prod was recently there.

The fix is a nested dict keyed by tenant ID. `_pheromone[tenant_id][node_name]`. Each tenant's colony evolves from its own workload history. Pheromone evaporation (ρ=0.05) and deposit (Q/score) operate per-tenant table. The `POST /reset` endpoint can clear a single tenant's trails without touching others — useful during demos, but also something you'd want in production when you onboard a new tenant whose first few jobs shouldn't inherit the priors of the tenants who came before.

> **Shared colony memory is the default. It looks right until you run multiple tenants. Then it's wrong in a way that's hard to attribute.**

---

## What the Dashboard Shows

The Grafana session starts cold. Every node sits near the pheromone floor — the colony has no history, so every node looks equally viable. Early placements are nearly random within the feasibility constraints.

By the 20th scheduling call, something shifts. The T4 node ($0.45/hr) starts accumulating trail. The V100 ($3.20/hr) sits idle. Not because any explicit cost rule is being applied — the `cost_efficiency` component of `CostEngine` handles that — but because the colony is reinforcing what's working. Good placements return low latency, high SLA headroom, low cost. Those nodes get deposit. Bad placements evaporate faster because the deposit is smaller. The divergence becomes self-reinforcing.

By the 35th call, the trails are stable. The pheromone bar chart in Grafana shows six bars of wildly different heights — two CPU nodes and the T4 dominating, the V100 and P100 close to floor. The colony has learned the topology from first principles.

The cost/job panel shows the same story with numbers. Early sessions: occasional V100 placements at $3.20/hr. Post-convergence: V100 is never selected for workloads where the T4 passes all constraints. The gap is roughly 7x per scheduling call, compounding across a multi-tenant session.

P99 latency: flat throughout. The ACO scoring loop, pheromone update, and Prometheus metric writes all fit inside the scheduling call window. The extender adds no perceptible latency.

---

## What's Still Missing

The spike prediction path is the one component that didn't behave as expected. Part 1 identified the problem: the Alibaba 2018 CPU trace has very few genuine spikes, and they're heavily smoothed. The LSTM sees them but can't distinguish them from ordinary variance. Spike recall in the benchmarks: 0%.

This isn't a model failure. The model is doing what it can with a dataset that doesn't have the signal it needs. The Alibaba 2022 GPU trace is the next dataset — it has sharper, more frequent spikes. The LSTM refit against that trace is the first item on the Part 3 roadmap.

Two others behind it: **node failure injection** (can the colony re-route when a highly-pheromone-weighted node suddenly goes offline? Or does it get stuck depositing on a dead trail?) and **Redis state store** (the pheromone tables currently live in-process Python dicts. A pod restart clears all colony memory. Redis changes that — the colony survives extender restarts and can be inspected, backed up, and replayed).

The repo is at [github.com/Aravind0403/ACO_Platform_Extension](https://github.com/Aravind0403/ACO_Platform_Extension). Full local demo in DEMO.md — no GKE required. The technical deep-dive on the extender protocol and what broke is on my [GitHub.io](https://aravind0403.github.io/).

I want to ask something specific: if you run multi-tenant scheduling in production — doesn't have to be ACO, any custom scheduler — how do you handle trail contamination between tenants? Do you isolate by namespace, by scoring weight, or do you just accept that tenant A's history affects tenant B and tune around it?

---

*Next: Spike prediction on the 2022 GPU trace, node failure injection, and moving the pheromone store off in-process dicts.*
