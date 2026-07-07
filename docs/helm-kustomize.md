# Helm and Kustomize Deployment

This guide covers deploying the application using Helm charts and Kustomize overlays.

---

## Deployment Methods

| Method | Best For | Complexity |
|--------|----------|------------|
| **Helm** (Primary) | Production, templating, releases | Medium |
| **Kustomize** (Alternative) | Environment overlays, GitOps | Low |

---

## Helm Deployment

Helm is a package manager for Kubernetes that uses charts to define, install, and upgrade applications.

### Chart Structure

```
infrastructure/backend/helm/fintech-simplifier/
├── Chart.yaml          # Chart metadata
├── values.yaml         # Default values
└── templates/
    ├── _helpers.tpl    # Template helpers
    ├── deployment.yaml # Deployment template
    ├── service.yaml    # Service template
    ├── configmap.yaml  # ConfigMap template
    ├── secrets.yaml    # Secrets template
    └── hpa.yaml        # HorizontalPodAutoscaler
```

### Install/Upgrade

```bash
# Basic install
helm upgrade --install fintech-simplifier \
  ./infrastructure/backend/helm/fintech-simplifier \
  --set image.repository=<ECR_REPOSITORY_URL> \
  --set image.tag=latest \
  --set secrets.nvidiaApiKey=nvapi-xxx

# With custom values file
helm upgrade --install fintech-simplifier \
  ./infrastructure/backend/helm/fintech-simplifier \
  -f custom-values.yaml
```

### Common Operations

```bash
# List releases
helm list

# Check release status
helm status fintech-simplifier

# View generated manifests (dry-run)
helm template fintech-simplifier ./infrastructure/backend/helm/fintech-simplifier

# Rollback to previous version
helm rollback fintech-simplifier 1

# Uninstall
helm uninstall fintech-simplifier
```

### Customizing Values

Create a `custom-values.yaml`:

```yaml
replicaCount: 3

image:
  repository: 123456789.dkr.ecr.ap-south-1.amazonaws.com/fintech-compliance
  tag: "v1.0.0"

resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
```

### Values Reference

| Value | Default | Description |
|-------|---------|-------------|
| `replicaCount` | 2 | Number of pod replicas |
| `image.repository` | "" | ECR repository URL |
| `image.tag` | "latest" | Image tag |
| `service.type` | LoadBalancer | Service type |
| `service.port` | 80 | External port |
| `resources.requests.memory` | 256Mi | Memory request |
| `resources.limits.memory` | 512Mi | Memory limit |
| `env.ENVIRONMENT` | production | Environment name |
| `env.NVIDIA_MODEL_NAME` | mistralai/mixtral-8x22b-instruct-v0.1 | LLM model |
| `secrets.nvidiaApiKey` | "" | NVIDIA API key |
| `autoscaling.enabled` | false | Enable HPA |

---

## Kustomize Deployment

Kustomize uses overlays to customize base manifests for different environments.

### Structure

```
infrastructure/backend/kustomize/
├── base/                    # Base resources
│   ├── kustomization.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
└── overlays/
    ├── production/          # Production patches
    │   ├── kustomization.yaml
    │   ├── deployment-patch.yaml
    │   └── configmap-patch.yaml
    └── staging/             # Staging patches
        ├── kustomization.yaml
        ├── deployment-patch.yaml
        └── configmap-patch.yaml
```

### Environment Differences

| Setting | Staging | Production |
|---------|---------|------------|
| Replicas | 1 | 3 |
| Memory Request | 256Mi | 512Mi |
| Memory Limit | 512Mi | 1Gi |
| DEBUG | true | false |

### Deploy with Kustomize

```bash
# Preview manifests
kustomize build infrastructure/backend/kustomize/overlays/production

# Apply to cluster
kubectl apply -k infrastructure/backend/kustomize/overlays/production

# Or directly with kustomize
kustomize build infrastructure/backend/kustomize/overlays/staging | kubectl apply -f -
```

### Update Image

```bash
cd infrastructure/backend/kustomize/overlays/production

# Set new image
kustomize edit set image IMAGE_PLACEHOLDER=<ECR_URL>:v1.0.0

# Apply
kubectl apply -k .
```

### Common Operations

```bash
# View final manifests
kustomize build overlays/production

# Diff against running state
kustomize build overlays/production | kubectl diff -f -

# Delete resources
kubectl delete -k overlays/production
```

---

## GitHub Actions Deployment

The CI/CD workflow supports Helm and Kustomize.

### Trigger Workflow

1. Go to **Actions** → **Deploy Backend to EKS**
2. Click **Run workflow**
3. Select:
   - **deploy_method**: `helm` (default) or `kustomize`
   - **environment**: `production` or `staging`

### Automatic Deployment

Push to `main` branch triggers Helm deployment automatically:

```bash
git push origin main
```

---

## Choosing a Method

### Use Helm when:
- Managing complex releases
- Need versioning and rollback
- Sharing charts across teams
- Templating with many variables

### Use Kustomize when:
- Environment-specific patches
- GitOps workflows (ArgoCD/Flux)
- Simple overlay requirements
- Prefer native kubectl tooling

