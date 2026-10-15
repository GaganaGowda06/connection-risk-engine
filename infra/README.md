# Infrastructure plan

This directory describes how the risk API would run on AWS. It has never been applied. There is
no AWS account behind this project, so it is checked with `terraform validate` and an offline
`terraform plan`, and nothing here holds a credential, key or account id.

## What would be created

A plan with the dummy values in the section below shows 16 resources:

- An ECR repository for the API image, with immutable tags, scan on push and a lifecycle rule
  that keeps the last 10 images.
- An S3 bucket for data and model artefacts, with versioning, AES256 encryption, all public
  access blocked, and a policy that refuses requests not made over TLS.
- A CloudWatch log group for the API with 14 day retention.
- Two IAM roles. The execution role may pull the API image, write to its own log group and
  nothing else. The task role may list and read the artefact bucket and nothing else.
- A security group that allows port 8000 from the ranges you give it and HTTPS out.
- An ECS cluster, a Fargate task definition for the API image (1 vCPU, 2 GB, read-only root
  filesystem, non-root user, health check on /health) and a service that runs it.

## What is not done yet

The container expects `data/` and `mlruns/` under `/app`, as it has in Docker Compose. Here
those are empty ephemeral volumes and the task role can read the bucket, but nothing copies the
artefacts from the bucket into the volumes at start. That step, and a load balancer in front of
the service, are needed before this serves traffic. The Darwin capture is not part of this plan.

## Checking it with no account

```
cd infra
terraform init -backend=false
terraform fmt -check
terraform validate
```

A plan needs some credential source to start the provider, even when it makes no AWS calls.
Setting `offline_plan = true` makes the provider skip credential validation, the account id
lookup, the metadata check and region validation. Put dummy values in a file that is not
committed (`infra/*.tfvars` is ignored) and give the provider throwaway environment variables:

```
cat > offline-plan.tfvars <<'VARS'
aws_region            = "eu-west-2"
vpc_id                = "vpc-00000000"
subnet_ids            = ["subnet-00000000", "subnet-11111111"]
allowed_ingress_cidrs = ["10.0.0.0/16"]
artefact_bucket_name  = "example-connection-risk-artefacts"
image_tag             = "offline"
offline_plan          = true
VARS
AWS_ACCESS_KEY_ID=offline AWS_SECRET_ACCESS_KEY=offline \
  terraform plan -var-file=offline-plan.tfvars
```

## Applying it with a real account

1. Sign in with credentials that can create the resources above, for example through AWS SSO.
2. Create a `prod.tfvars` with your region, VPC, subnets, ingress ranges and a globally unique
   bucket name. Leave `offline_plan` false.
3. Run `terraform init`, then `terraform plan -var-file=prod.tfvars` and read it.
4. Apply once to create the repository, push the API image to the `ecr_repository_url` output
   with the tag you set as `image_tag`, then apply again so the service can start.
5. Add remote state (an S3 backend with locking) before more than one person applies. State is
   local in this directory, and is ignored by git.

## Costs

Costs were not estimated. Fargate, CloudWatch logs, S3 storage and requests, ECR storage and any
NAT gateway or load balancer in front of the service would all be billed, and no figure here has
been checked against the AWS price list.
