terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ── GKE Cluster ───────────────────────────────────────────────────────────────
#
# We use a single zonal cluster for cost control.
# The default node pool is deleted immediately — we manage pools explicitly.
#
resource "google_container_cluster" "aco" {
  name     = var.cluster_name
  location = var.zone

  # Remove the default node pool — we define our own below
  remove_default_node_pool = true
  initial_node_count       = 1

  # Workload identity lets pods call GCP APIs without service account key files
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Logging / monitoring to Cloud Operations (free tier covers demo usage)
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"
}

# ── CPU node pool (default — no GPU) ─────────────────────────────────────────
#
# n2-standard-4: 4 vCPU, 16 GB RAM.
# Used for latency-critical and stream-processing workloads.
# CostEngine labels: aco/gpu-type=none, aco/cost-per-hour=0.19
#
resource "google_container_node_pool" "cpu" {
  name       = "cpu-pool"
  cluster    = google_container_cluster.aco.name
  location   = var.zone
  node_count = var.cpu_node_count

  node_config {
    machine_type = "n2-standard-4"
    disk_size_gb = 50
    disk_type    = "pd-ssd"
    image_type   = "COS_CONTAINERD"

    # Labels — the extender reads these to build ComputeNode objects
    labels = {
      "aco/gpu-type"       = "none"
      "aco/instance-type"  = "on_demand"
      "aco/cost-per-hour"  = "0.19"
      "aco/arch"           = "x86_64"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}

# ── T4 GPU node pool (cheapest GPU tier) ──────────────────────────────────────
#
# n1-standard-4 + 1x NVIDIA T4: good for inference, batch ML.
# ~$0.45/hr on-demand. The ACO CostEngine will prefer this for batch GPU jobs
# once pheromone converges (lowest cost → highest cost_efficiency_factor).
#
resource "google_container_node_pool" "t4" {
  name       = "t4-pool"
  cluster    = google_container_cluster.aco.name
  location   = var.zone
  node_count = var.t4_node_count

  node_config {
    machine_type = "n1-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    image_type   = "COS_CONTAINERD"

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    labels = {
      "aco/gpu-type"       = "t4"
      "aco/gpu-count"      = "1"
      "aco/instance-type"  = "on_demand"
      "aco/cost-per-hour"  = "0.45"
      "aco/arch"           = "x86_64"
    }

    # GPU nodes need this taint so only GPU-requesting pods land here
    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}

# ── A10 GPU node pool (mid-tier) ──────────────────────────────────────────────
#
# a2-highgpu-1g + 1x NVIDIA A10: modern ampere GPU, good for training.
# ~$1.20/hr on-demand. ACO will route here when T4 is saturated or
# the job priority is high enough to justify the cost.
#
resource "google_container_node_pool" "a10" {
  name       = "a10-pool"
  cluster    = google_container_cluster.aco.name
  location   = var.zone
  node_count = var.a10_node_count

  node_config {
    machine_type = "a2-highgpu-1g"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    image_type   = "COS_CONTAINERD"

    guest_accelerator {
      type  = "nvidia-tesla-a100"   # A10 maps to a100 family on GCP
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    labels = {
      "aco/gpu-type"       = "a10"
      "aco/gpu-count"      = "1"
      "aco/instance-type"  = "on_demand"
      "aco/cost-per-hour"  = "1.20"
      "aco/arch"           = "x86_64"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}

# ── V100 GPU node pool (high-tier) ────────────────────────────────────────────
#
# n1-standard-8 + 1x NVIDIA V100: high-memory training GPU.
# ~$2.48/hr on-demand. ACO will deprioritise this for batch jobs due to
# cost_efficiency_factor — only routes here for high-priority (priority > 80)
# or latency-critical GPU jobs.
#
resource "google_container_node_pool" "v100" {
  name       = "v100-pool"
  cluster    = google_container_cluster.aco.name
  location   = var.zone
  node_count = var.v100_node_count

  node_config {
    machine_type = "n1-standard-8"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    image_type   = "COS_CONTAINERD"

    guest_accelerator {
      type  = "nvidia-tesla-v100"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    labels = {
      "aco/gpu-type"       = "v100"
      "aco/gpu-count"      = "1"
      "aco/instance-type"  = "on_demand"
      "aco/cost-per-hour"  = "2.48"
      "aco/arch"           = "x86_64"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}

# ── P100 GPU node pool (legacy high-tier) ─────────────────────────────────────
#
# n1-standard-8 + 1x NVIDIA P100: older Volta GPU.
# ~$1.60/hr — sits between A10 and V100 in cost.
# Useful for showing ACO routing around a cost-tier gap.
#
resource "google_container_node_pool" "p100" {
  name       = "p100-pool"
  cluster    = google_container_cluster.aco.name
  location   = var.zone
  node_count = var.p100_node_count

  node_config {
    machine_type = "n1-standard-8"
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    image_type   = "COS_CONTAINERD"

    guest_accelerator {
      type  = "nvidia-tesla-p100"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    labels = {
      "aco/gpu-type"       = "p100"
      "aco/gpu-count"      = "1"
      "aco/instance-type"  = "on_demand"
      "aco/cost-per-hour"  = "1.60"
      "aco/arch"           = "x86_64"
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}
