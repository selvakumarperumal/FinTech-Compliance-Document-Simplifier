# Deployment Guide

Step-by-step guide to deploy the FinTech Compliance Document Simplifier to AWS EKS.

## Overview

The deployment consists of three phases:

1. **Bootstrap** - Create state management (S3 + DynamoDB)
2. **Infrastructure** - Deploy VPC, EKS, ECR via Terraform
3. **Application** - Deploy via Helm (primary) or Kustomize

---

## Phase 1: Bootstrap State Management

Create S3 bucket and DynamoDB table for Terraform state.

```bash
cd infrastructure/bootstrap

terraform init
terraform plan
terraform apply
```

**Outputs:**
```
state_bucket_name = "fintech-compliance-terraform-state"
dynamodb_table_name = "fintech-compliance-terraform-locks"
```

### Update backend.hcl

Verify `infrastructure/backend.hcl` has correct values:

```hcl
bucket         = "fintech-compliance-terraform-state"
key            = "backend/terraform.tfstate"
region         = "ap-south-1"
dynamodb_table = "fintech-compliance-terraform-locks"
encrypt        = true
```

---

## Phase 2: Deploy Infrastructure

Deploy VPC, EKS cluster, and ECR repository.

```bash
cd infrastructure/backend

terraform init -backend-config=../backend.hcl
terraform plan
terraform apply
```

> ⏱️ EKS cluster creation takes 15-20 minutes.

**Outputs:**
```
cluster_endpoint = "https://xxx.eks.amazonaws.com"
cluster_name = "fintech-compliance"
ecr_repository_url = "123456789.dkr.ecr.ap-south-1.amazonaws.com/fintech-compliance"
```

### Configure kubectl

```bash
aws eks update-kubeconfig --region ap-south-1 --name fintech-compliance
kubectl get nodes
```

---

## Phase 3: Deploy Application

### Method A: Helm (Recommended)

#### 1. Build and Push Docker Image

```bash
cd backend

# Login to ECR
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin <ECR_URL>

# Build and push
docker build -t fintech-compliance .
docker tag fintech-compliance:latest <ECR_URL>:latest
docker push <ECR_URL>:latest
```

#### 2. Deploy with Helm

```bash
helm upgrade --install fintech-simplifier \
  ./infrastructure/backend/helm/fintech-simplifier \
  --set image.repository=<ECR_URL> \
  --set image.tag=latest \
  --set secrets.nvidiaApiKey=nvapi-xxx
```

#### 3. Verify Deployment

```bash
helm status fintech-simplifier
kubectl get pods
kubectl get svc
```

---

### Method B: Kustomize (Alternative)

#### 1. Build and Push Image (same as above)

#### 2. Create Secret

```bash
kubectl create secret generic fintech-secrets \
  --from-literal=NVIDIA_API_KEY=nvapi-xxx
```

#### 3. Deploy with Kustomize

```bash
cd infrastructure/backend/kustomize/overlays/production

# Set image
kustomize edit set image IMAGE_PLACEHOLDER=<ECR_URL>:latest

# Apply
kubectl apply -k .
```

---

### Method C: GitHub Actions (CI/CD)

Push to `main` branch triggers automatic deployment:

```bash
git add .
git commit -m "Deploy update"
git push origin main
```

Or trigger manually via Actions → Deploy Backend to EKS.

---

## Verify Deployment

### Check Pods

```bash
kubectl get pods -l app.kubernetes.io/name=fintech-simplifier
```

### Get Service URL

```bash
kubectl get svc fintech-simplifier -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

### Test API

```bash
curl http://<LOAD_BALANCER_URL>/docs
```

---

## Environment-Specific Deployment

### Staging

```bash
# Helm
helm upgrade --install fintech-simplifier-staging \
  ./infrastructure/backend/helm/fintech-simplifier \
  --set env.ENVIRONMENT=staging \
  --set env.DEBUG=true \
  --set replicaCount=1

# Kustomize
kubectl apply -k infrastructure/backend/kustomize/overlays/staging
```

### Production

```bash
# Helm (default)
helm upgrade --install fintech-simplifier \
  ./infrastructure/backend/helm/fintech-simplifier

# Kustomize
kubectl apply -k infrastructure/backend/kustomize/overlays/production
```

---

## Cleanup

### Delete Application

```bash
# Helm
helm uninstall fintech-simplifier

# Kustomize
kubectl delete -k infrastructure/backend/kustomize/overlays/production
```

### Destroy Infrastructure

```bash
cd infrastructure/backend
terraform destroy

cd ../bootstrap
terraform destroy  # Optional: keeps state history
```

> ⚠️ This deletes all resources including the EKS cluster.
