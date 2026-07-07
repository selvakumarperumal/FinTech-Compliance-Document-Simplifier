# Terraform Resources

Detailed documentation of all Terraform resources used in this infrastructure.

---

## Bootstrap Module

Located in `infrastructure/bootstrap/`

### Resources

#### aws_s3_bucket.terraform_state

S3 bucket for storing Terraform state files.

| Attribute | Value |
|-----------|-------|
| Bucket Name | `fintech-compliance-terraform-state` |
| Versioning | Enabled |
| Encryption | AES256 |
| Public Access | Blocked |

#### aws_dynamodb_table.terraform_locks

DynamoDB table for state locking to prevent concurrent modifications.

| Attribute | Value |
|-----------|-------|
| Table Name | `fintech-compliance-terraform-locks` |
| Billing Mode | PAY_PER_REQUEST |
| Hash Key | `LockID` (String) |

---

## Backend Module

Located in `infrastructure/backend/`

### Providers

```hcl
# AWS Provider
provider "aws" {
  region = "ap-south-1"
}

# Kubernetes Provider (configured via EKS)
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
}
```

---

### VPC Resources (vpc.tf)

Uses the `terraform-aws-modules/vpc/aws` module.

#### Network Layout

| Subnet Type | CIDR Blocks | Purpose |
|-------------|-------------|---------|
| VPC | 10.0.0.0/16 | Main VPC |
| Public | 10.0.101.0/24, 10.0.102.0/24, 10.0.103.0/24 | Load Balancers, NAT Gateway |
| Private | 10.0.1.0/24, 10.0.2.0/24, 10.0.3.0/24 | EKS Worker Nodes |

#### Key Features

- **NAT Gateway**: Single NAT gateway for cost optimization
- **DNS**: Hostnames and support enabled
- **Subnet Tags**: Properly tagged for EKS integration

```hcl
public_subnet_tags = {
  "kubernetes.io/role/elb" = 1
  "kubernetes.io/cluster/fintech-compliance" = "shared"
}

private_subnet_tags = {
  "kubernetes.io/role/internal-elb" = 1
  "kubernetes.io/cluster/fintech-compliance" = "shared"
}
```

---

### EKS Cluster (eks.tf)

Uses the `terraform-aws-modules/eks/aws` module.

#### Cluster Configuration

| Parameter | Value |
|-----------|-------|
| Cluster Name | `fintech-compliance` |
| Kubernetes Version | 1.28 |
| Endpoint Access | Public |
| Subnets | Private subnets only |

#### Node Group Configuration

| Parameter | Value |
|-----------|-------|
| Instance Type | t3.medium |
| Min Nodes | 1 |
| Max Nodes | 4 |
| Desired Nodes | 2 |
| AMI Type | AL2_x86_64 (Amazon Linux 2) |

#### IAM Roles Created

- **Cluster Role**: Allows EKS to manage AWS resources
- **Node Role**: Allows worker nodes to join cluster
- **OIDC Provider**: For IAM Roles for Service Accounts (IRSA)

---

### ECR Repository (ecr.tf)

#### aws_ecr_repository.app

Container registry for Docker images.

| Attribute | Value |
|-----------|-------|
| Repository Name | `fintech-compliance` |
| Image Tag Mutability | MUTABLE |
| Scan on Push | Enabled |

#### aws_ecr_lifecycle_policy.app

Automatic cleanup of old images.

| Rule | Action |
|------|--------|
| Keep last 10 images | Expire older images |

---

## Variables Reference

### Required Variables

None - all have defaults.

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `aws_region` | `ap-south-1` | AWS region |
| `project_name` | `fintech-compliance` | Project name for resources |
| `environment` | `production` | Environment tag |
| `eks_cluster_version` | `1.28` | Kubernetes version |
| `eks_node_instance_type` | `t3.medium` | EC2 instance type |
| `eks_desired_nodes` | `2` | Desired node count |
| `eks_min_nodes` | `1` | Minimum nodes |
| `eks_max_nodes` | `4` | Maximum nodes |
| `container_port` | `8000` | Application port |

### Customizing Variables

Create `terraform.tfvars`:

```hcl
aws_region             = "us-east-1"
eks_node_instance_type = "t3.large"
eks_desired_nodes      = 3
```

---

## Outputs Reference

| Output | Description |
|--------|-------------|
| `cluster_endpoint` | EKS API server endpoint |
| `cluster_name` | EKS cluster name |
| `ecr_repository_url` | ECR repository URL for Docker images |
| `kubeconfig_command` | Command to configure kubectl |
| `load_balancer_hostname` | Instructions to get LB hostname |

### Accessing Outputs

```bash
terraform output cluster_endpoint
terraform output -json | jq
```
