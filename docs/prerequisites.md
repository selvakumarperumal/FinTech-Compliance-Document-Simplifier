# Prerequisites

Before deploying the infrastructure, ensure you have the following set up.

## Required Tools

| Tool | Version | Installation |
|------|---------|--------------|
| AWS CLI | >= 2.0 | [Install Guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| Terraform | >= 1.0 | [Install Guide](https://developer.hashicorp.com/terraform/downloads) |
| kubectl | >= 1.28 | [Install Guide](https://kubernetes.io/docs/tasks/tools/) |
| Docker | >= 24.0 | [Install Guide](https://docs.docker.com/get-docker/) |

### Verify Installations

```bash
# Check AWS CLI
aws --version

# Check Terraform
terraform --version

# Check kubectl
kubectl version --client

# Check Docker
docker --version
```

## AWS Account Setup

### 1. Create IAM User

Create an IAM user with programmatic access and the following policies:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:*",
        "eks:*",
        "ecr:*",
        "s3:*",
        "dynamodb:*",
        "iam:*",
        "elasticloadbalancing:*",
        "logs:*",
        "autoscaling:*"
      ],
      "Resource": "*"
    }
  ]
}
```

> **Note**: For production, use more restrictive policies.

### 2. Configure AWS CLI

```bash
aws configure
```

Enter:
- AWS Access Key ID
- AWS Secret Access Key
- Default region: `ap-south-1`
- Default output format: `json`

### 3. Verify AWS Access

```bash
aws sts get-caller-identity
```

## GitHub Secrets

For CI/CD pipeline, add these secrets to your GitHub repository:

| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `NVIDIA_API_KEY` | NVIDIA API key for LLM |

### Adding Secrets

1. Go to repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each secret

## NVIDIA API Key

1. Create account at [build.nvidia.com](https://build.nvidia.com)
2. Select a model (e.g., Mixtral 8x22B)
3. Click "Get API Key" → "Generate Key"
4. Save the key (starts with `nvapi-`)

## Local Environment

Create `.env` file in project root:

```bash
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_MODEL_NAME=mistralai/mixtral-8x22b-instruct-v0.1
```

## Cost Considerations

Estimated monthly costs for this infrastructure:

| Resource | Estimated Cost |
|----------|----------------|
| EKS Cluster | ~$73/month |
| EC2 (2x t3.medium) | ~$60/month |
| NAT Gateway | ~$32/month |
| Load Balancer | ~$16/month |
| ECR Storage | ~$1/month |
| S3 + DynamoDB | ~$1/month |
| **Total** | **~$183/month** |

> Costs vary by region and usage. Use AWS Cost Calculator for accurate estimates.
