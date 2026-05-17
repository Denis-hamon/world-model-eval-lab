# =============================================================================
# wmel GPU experiment on OVH AI Training
# =============================================================================
#
# What this does:
#   1. Generates a unique job name (prefix + 4-char random suffix) so re-running
#      `terraform apply` does not collide with a previous job.
#   2. Submits an AI Training job via the `ovhai` CLI (called from a local-exec
#      provisioner). The job clones world-model-eval-lab, installs extras, and
#      runs the experiment specified in var.experiment_command.
#
# Why ovhai CLI rather than a native Terraform resource:
#   OVH's Terraform provider coverage for AI Training has been spotty across
#   versions. The CLI is the stable, supported surface. local-exec gives us a
#   clean way to call it and tie the lifecycle to a Terraform apply.
#
# What this does NOT manage (intentionally v0):
#   - Persistent output datastore (Swift container + S3 credentials). For a
#     smoke test the job logs are captured by `ovhai job logs` and the JSON
#     dump prints to stdout. For real training campaigns you will want to
#     pre-create a Swift container manually and add `--volume <ctr>@<region>:
#     /workspace/output:RW` to the ovhai call below.
#   - Job cancellation on `terraform destroy`. Use `ovhai job kill` manually
#     (the outputs print the exact command).
#   - Quota requests. If your account has 0 A100 quota, this module cannot
#     fix it. Open an OVH support ticket.

provider "ovh" {
  endpoint           = var.ovh_endpoint
  application_key    = var.ovh_application_key
  application_secret = var.ovh_application_secret
  consumer_key       = var.ovh_consumer_key
}

# Random suffix so successive applies produce distinct job names.
resource "random_id" "suffix" {
  byte_length = 2
}

locals {
  job_name   = "${var.job_name_prefix}-${random_id.suffix.hex}"
  job_script = file("${path.module}/job_train.sh")
}

# Submit the AI Training job via the ovhai CLI. The null_resource trigger
# pattern below re-runs the provisioner whenever the script, image, flavor,
# experiment command, or git ref changes — so editing job_train.sh and
# re-applying re-submits a fresh job.
resource "null_resource" "submit_job" {
  triggers = {
    job_script_sha = sha1(local.job_script)
    docker_image   = var.docker_image
    gpu_flavor     = var.gpu_flavor
    gpu_count      = var.gpu_count
    experiment     = var.experiment_command
    wmel_ref       = var.wmel_git_ref
    job_name       = local.job_name
  }

  provisioner "local-exec" {
    command = <<-EOT
      set -e
      ovhai job run \
        --name "${local.job_name}" \
        --flavor "${var.gpu_flavor}" \
        --gpu ${var.gpu_count} \
        --timeout ${var.job_timeout_seconds} \
        --label wmel=true \
        --label managed-by=terraform \
        --envvar WMEL_GIT_REF=${var.wmel_git_ref} \
        --envvar EXPERIMENT_CMD=${var.experiment_command} \
        ${var.docker_image} \
        -- bash -c "${replace(local.job_script, "\"", "\\\"")}"
    EOT
  }
}
