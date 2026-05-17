# `infra/terraform/` — spawn a GPU AI Training job on OVH

Submits a `world-model-eval-lab` experiment as an OVH AI Training job. Tied
to a single OVH Public Cloud project. Reusable: `terraform apply` again with
a different `experiment_command` or `wmel_git_ref` and you get a fresh job.

This is a **v0**: just job submission. No managed S3 datastore yet (see
"Datastore" below for the manual workaround until that lands).

## What this manages

- A unique job name (`<prefix>-<random>`) so successive applies do not
  collide.
- An AI Training job submission via the `ovhai` CLI, wrapped in a
  `null_resource` so Terraform tracks its lifecycle and re-submits when any
  of the script, image, flavor, GPU count, or `experiment_command` changes.

## What this does NOT manage

- **Persistent output storage.** For a smoke test, job logs are captured by
  `ovhai job logs` and the experiment dumps JSON to stdout. For real
  training campaigns you will want to pre-create a Swift container manually
  (see "Datastore" below).
- **Job cancellation on destroy.** `terraform destroy` removes the
  `null_resource` from state; it does not kill the running job. Use
  `ovhai job kill <job_name>` (the Terraform outputs print the exact
  command).
- **Quota.** If your account has 0 A100 quota, no apply will fix that. Open
  a ticket at https://help.ovhcloud.com — title:
  *"Demande d'augmentation de quota GPU A100 pour AI Training"*.
- **Cost.** A running A100 instance bills per second. The
  `job_timeout_seconds` variable (default 12h) is your hard kill switch;
  this is still your money.

## Prerequisites

1. Terraform >= 1.6.
2. `ovhai` CLI installed and authenticated on the machine running Terraform:
   ```bash
   pip install ovh-ai-cli
   ovhai login                          # one-time OAuth in browser
   ovhai capabilities flavor list       # confirm your GPU flavor + quota
   ```
3. OVH API credentials (application key + secret + consumer key) generated
   at https://api.ovh.com/createToken, with the following minimum rights:
   - `GET /cloud/project/*`
   - `POST /cloud/project/*` (the provider initialisation reads project info)
4. Your OVH Public Cloud project ID — the UUID-ish slug from the manager URL
   at https://www.ovh.com/manager/#/public-cloud/pci/projects.

## Usage

```bash
cd infra/terraform/

# 1. Credentials go in environment variables, never in any file.
export OVH_ENDPOINT="ovh-eu"
export OVH_APPLICATION_KEY="..."
export OVH_APPLICATION_SECRET="..."
export OVH_CONSUMER_KEY="..."

# 2. Non-secret variables go in terraform.tfvars (gitignored).
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars: set ovh_service_name, gpu_flavor, experiment_command.

# 3. Plan then apply.
terraform init
terraform plan -out=plan.tfplan
terraform apply plan.tfplan

# 4. Stream the job logs.
$(terraform output -raw follow_logs_command)
# or one shot:
ovhai job get $(terraform output -raw job_name)
```

## Defaults are a smoke test, not a real GPU workload

`experiment_command` defaults to `experiments.dmc_acrobot.cpg`, the v0.11 CPG
run on Acrobot. **This is CPU-bound and burns the A100 for no useful gain.**
It is the default on purpose: it validates the entire pipeline end-to-end
(`ovhai` reaches OVH, image pulls, repo clones, deps install, CUDA is
visible, `dm_control` loads, results dump) at a low cost.

Once you trust the pipeline, override `experiment_command` with a real GPU
workload — a TD-MPC2 training entry point once the adapter lands in
`src/wmel/adapters/`.

## Datastore (manual until v1)

To make `/workspace/output` inside the job persist to S3, do this once:

```bash
# 1. Create a Swift container in your region.
ovhai data upload-init <region>
# (outputs a container name like 'datastore-XXX')

# 2. Add a volume mount to the ovhai call in main.tf, after --label lines:
#     --volume <container_name>@<region>:/workspace/output:RW \
#
# 3. terraform apply. The job script already cp's results/ into
#    /workspace/output when the mount is present.
```

## Credentials hygiene

- **Never** paste credentials into chat, into `terraform.tfvars`, or any
  file under version control. Environment variables in your shell are the
  only intended sink.
- If a credential leaked (a token pasted into a chat, an application key
  emailed to yourself, a tfvars accidentally `git add`-ed), revoke it
  immediately at https://www.ovh.com/auth/api/listToken and regenerate.
- The `.gitignore` shipped here keeps `terraform.tfstate` and
  `*.tfvars` (except `.example`) out of git. Treat your state file as a
  secret regardless — it can hold sensitive output values if you add
  managed datastore resources later.

## Cleanup

```bash
ovhai job kill $(terraform output -raw job_name)   # if still running
terraform destroy                                  # frees Terraform state
```
