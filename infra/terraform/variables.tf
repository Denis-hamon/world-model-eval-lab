# All credentials below default to environment variables so terraform.tfvars
# does not need to hold them. Set in your shell:
#
#   export OVH_ENDPOINT="ovh-eu"
#   export OVH_APPLICATION_KEY="..."
#   export OVH_APPLICATION_SECRET="..."
#   export OVH_CONSUMER_KEY="..."
#   export TF_VAR_ovh_service_name="..."   # your Public Cloud project ID
#
# Never paste credentials into terraform.tfvars that gets committed.

variable "ovh_endpoint" {
  description = "OVH API endpoint (ovh-eu, ovh-us, ovh-ca, kimsufi-eu, ...). Defaults to ovh-eu."
  type        = string
  default     = "ovh-eu"
}

variable "ovh_application_key" {
  description = "OVH API application key. Prefer setting via env: OVH_APPLICATION_KEY."
  type        = string
  sensitive   = true
  default     = null
}

variable "ovh_application_secret" {
  description = "OVH API application secret. Prefer setting via env: OVH_APPLICATION_SECRET."
  type        = string
  sensitive   = true
  default     = null
}

variable "ovh_consumer_key" {
  description = "OVH API consumer key. Prefer setting via env: OVH_CONSUMER_KEY."
  type        = string
  sensitive   = true
  default     = null
}

variable "ovh_service_name" {
  description = "OVH Public Cloud project ID. Find it in Manager > Cloud > Public Cloud > project URL slug. Set via env: TF_VAR_ovh_service_name."
  type        = string
}

variable "region" {
  description = "OVH region for the AI Training job and the S3 datastore. GRA / SBG / BHS / WAW / DE / UK."
  type        = string
  default     = "GRA"
}

variable "gpu_flavor" {
  description = "Instance flavor. Examples: ai1-1-a100-80, ai1-1-v100s, ai1-1-h100, ai1-1-l40s, ai1-1-l4. Verify with `ovhai capabilities flavor list`."
  type        = string
  default     = "ai1-1-a100-80"
}

variable "gpu_count" {
  description = "Number of GPUs to attach. World-model training on DMC fits in 1."
  type        = number
  default     = 1
}

variable "docker_image" {
  description = "OVH pre-built image. PyTorch + JupyterLab works for TD-MPC2 and Dreamer-V3 (PyTorch port)."
  type        = string
  default     = "ovhcom/ai-training-pytorch:latest"
}

variable "job_name_prefix" {
  description = "Prefix for the job name. A random 4-char suffix is appended so reruns do not collide."
  type        = string
  default     = "wmel-gpu"
}

variable "job_timeout_seconds" {
  description = "Hard kill switch on the job. Default: 12 hours."
  type        = number
  default     = 43200
}

variable "wmel_git_ref" {
  description = "Branch, tag, or commit SHA of world-model-eval-lab to check out inside the container."
  type        = string
  default     = "main"
}

variable "experiment_command" {
  description = <<-EOT
    The Python module path (after wmel install) the job runs. Default reproduces the v0.11 CPG run on DMC Acrobot,
    which is CPU-bound and is intended as a smoke test of the pipeline (CUDA available, dm-control loads, S3 writeback works).
    For an actual GPU-burning experiment, override with a TD-MPC2 / Dreamer-V3 training entry point once an adapter is committed.
  EOT
  type        = string
  default     = "experiments.dmc_acrobot.cpg"
}
