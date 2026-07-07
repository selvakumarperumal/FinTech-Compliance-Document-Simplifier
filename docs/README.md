# Infrastructure Documentation

This documentation covers the AWS infrastructure setup for the FinTech Compliance Document Simplifier.

## Table of Contents

1. [Architecture Overview](./architecture.md)
2. [Prerequisites](./prerequisites.md)
3. [Deployment Guide](./deployment-guide.md)
4. [Terraform Resources](./terraform-resources.md)
5. [Helm & Kustomize](./helm-kustomize.md)
6. [CI/CD Pipeline](./cicd.md)
7. [Troubleshooting](./troubleshooting.md)

## Quick Start

```bash
# 1. Bootstrap state management
cd infrastructure/bootstrap
terraform init && terraform apply

# 2. Deploy infrastructure
cd ../backend
terraform init -backend-config=../backend.hcl && terraform apply

# 3. Configure kubectl
aws eks update-kubeconfig --region ap-south-1 --name fintech-compliance

# 4. Deploy application with Helm (recommended)
helm upgrade --install fintech-simplifier ./infrastructure/backend/helm/fintech-simplifier \
  --set image.repository=<ECR_URL> \
  --set secrets.nvidiaApiKey=<KEY>

# Alternative: Deploy with Kustomize
kubectl apply -k infrastructure/backend/kustomize/overlays/production
```

## Directory Structure

```
infrastructure/
├── backend.hcl             # Backend state configuration
├── bootstrap/              # State management (S3 + DynamoDB)
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
└── backend/                # Main infrastructure
    ├── main.tf             # Providers configuration
    ├── variables.tf        # Input variables
    ├── outputs.tf          # Output values
    ├── vpc.tf              # VPC and networking
    ├── eks.tf              # EKS cluster
    ├── ecr.tf              # Container registry
    ├── helm/               # Helm charts (primary)
    │   └── fintech-simplifier/
    │       ├── Chart.yaml
    │       ├── values.yaml
    │       └── templates/
    └── kustomize/          # Kustomize overlays (alternative)
        ├── base/
        └── overlays/
            ├── production/
            └── staging/
```


