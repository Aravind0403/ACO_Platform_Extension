---
layout: about
title: About
permalink: /about/
---

I work on distributed systems, platform infrastructure, and reliability engineering. Seven years across Microsoft R&D India and Amazon Development Centre, building the kind of systems that are boring in the best possible way — ones that handle load quietly and fail gracefully.

My research sits at the boundary between scheduling theory and what the infrastructure can actually do at runtime.

**[ACO-Adaptive](https://github.com/Aravind0403/ACO_Platform_Extension)** is a cluster scheduler combining Ant Colony Optimisation with a per-node predictor for cost-aware, QoS-preserving GPU and CPU job placement. A 14-way ablation across LSTM, GRU, TCN, ARIMA, EMA, and others found that EMA (α=0.5) outperformed every trained model on safe-node routing — 89.2% vs 75.4% for LSTM, at zero training cost. The paper is under review at a peer-reviewed systems conference. The platform extension wires the validated algorithm into a Kubernetes scheduler extender with Prometheus observability and Terraform-provisioned GPU node pools.

**[Clairvoyant](https://arxiv.org/abs/2606.07248)** is a drop-in scheduling proxy for serial LLM backends — Ollama, llama.cpp — that reduces short-request P50 latency by 70–76% under burst conditions without modifying the backend. The core finding is an empirical inversion: under FCFS, short requests accumulate higher queue-wait P50 (20.28s) than long requests (15.20s), despite shorter service time — a direct signature of head-of-line blocking. On hardware where KV-cache continuous batching isn't feasible, the admission queue is the only schedulable point; Clairvoyant exploits it with a 19-feature XGBoost classifier running at 0.029ms P99.

This blog is where I write about what the benchmarks miss.

---

*GitHub: [Aravind0403](https://github.com/Aravind0403)*  
*arXiv: [2606.07248](https://arxiv.org/abs/2606.07248)*  
*Email: aravindsharma20@gmail.com*
