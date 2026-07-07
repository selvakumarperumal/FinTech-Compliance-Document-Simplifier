# CI/CD Pipeline

Documentation for the GitHub Actions deployment pipeline.

---

## Overview

The CI/CD pipeline automates build and deployment using **Helm** (primary) or **Kustomize**:

```
Push to main → Build Image → Push to ECR → Deploy via Helm/Kustomize
```

---

## Workflows

| Workflow | Purpose | Trigger |
|----------|---------|---------|
| `deploy-backend.yml` | Build and deploy application | Push to main, Manual |
| `infrastructure.yml` | Manage Terraform infrastructure | Manual only |

---

## Deploy Backend Workflow

Located at `.github/workflows/deploy-backend.yml`

### Trigger Conditions

| Trigger | Description |
|---------|-------------|
| Push to `main` (backend/**) | Auto-deploy with Helm |
| Manual (workflow_dispatch) | Choose Helm or Kustomize |

### Jobs

```
┌──────────────────┐     ┌─────────────────┐
│  build-and-push  │────►│   deploy-helm   │
│  (Docker image)  │     │ (if helm)       │
└──────────────────┘     └─────────────────┘
                              │
                              ▼
                         ┌─────────────────┐
                         │deploy-kustomize │
                         │ (if kustomize)  │
                         └─────────────────┘
```

### Deployment Options

When triggering manually, select:

| Option | Values |
|--------|--------|
| `deploy_method` | `helm` (default), `kustomize` |
| `environment` | `production` (default), `staging` |

### Helm Deployment

```yaml
helm upgrade --install fintech-simplifier \
  ./infrastructure/backend/helm/fintech-simplifier \
  --set image.repository=$ECR_REGISTRY/$ECR_REPOSITORY \
  --set image.tag=$IMAGE_TAG \
  --set secrets.nvidiaApiKey=${{ secrets.NVIDIA_API_KEY }}
```

### Kustomize Deployment

```yaml
cd infrastructure/backend/kustomize/overlays/$ENVIRONMENT
kustomize edit set image IMAGE_PLACEHOLDER=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
kustomize build . | kubectl apply -f -
```

---

## Infrastructure Workflow

Located at `.github/workflows/infrastructure.yml`

### Actions

| Action | Description |
|--------|-------------|
| `plan` | Preview Terraform changes |
| `apply` | Apply infrastructure changes |
| `destroy` | Destroy all resources |

### Usage

1. Go to **Actions** → **Infrastructure Management**
2. Click **Run workflow**
3. Select action (`plan`, `apply`, `destroy`)
4. Enable `auto_approve` for destroy (required)

---

## Required Secrets

Add to GitHub → Settings → Secrets → Actions:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `NVIDIA_API_KEY` | NVIDIA API key for LLM |

---

## Manual Trigger

1. Go to repository → **Actions**
2. Select workflow
3. Click **Run workflow**
4. Configure options and run

---

## Monitoring

### View Deployment Status

```bash
# Helm releases
helm list

# Pods
kubectl get pods -l app.kubernetes.io/name=fintech-simplifier

# Service URL
kubectl get svc fintech-simplifier
```

### View Logs

```bash
kubectl logs -l app.kubernetes.io/name=fintech-simplifier -f
```

---

## Rollback

### Helm Rollback

```bash
# View history
helm history fintech-simplifier

# Rollback to previous
helm rollback fintech-simplifier

# Rollback to specific revision
helm rollback fintech-simplifier 2
```

### Kustomize Rollback

```bash
kubectl rollout undo deployment -l app=fintech-simplifier
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check Dockerfile and dependencies |
| ECR push fails | Verify AWS credentials |
| Helm deploy fails | Check `helm status fintech-simplifier` |
| Kustomize fails | Validate with `kustomize build` |
| Pods not starting | Check `kubectl describe pod` |
