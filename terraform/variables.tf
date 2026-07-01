variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for the cluster"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for zonal node pools"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "aco-platform"
}

# ── Node pool sizing ──────────────────────────────────────────────────────────
# Keep counts at 1 for demo / cost control.
# Bump to 2–3 for load testing or the live HiPC demo.

variable "cpu_node_count" {
  description = "Number of nodes in the CPU (default) node pool"
  type        = number
  default     = 2
}

variable "t4_node_count" {
  description = "Number of T4 GPU nodes (cheapest GPU tier — $0.45/hr)"
  type        = number
  default     = 1
}

variable "a10_node_count" {
  description = "Number of A10 GPU nodes (mid-tier — $1.20/hr)"
  type        = number
  default     = 1
}

variable "v100_node_count" {
  description = "Number of V100 GPU nodes (high-tier — $2.48/hr)"
  type        = number
  default     = 1
}

variable "p100_node_count" {
  description = "Number of P100 GPU nodes (legacy high-tier — $1.60/hr)"
  type        = number
  default     = 1
}
