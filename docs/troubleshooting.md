# Troubleshooting Guide

Common issues and solutions for the infrastructure.

---

## Terraform Issues

### Error: Backend Initialization Required

```
Error: Backend initialization required, please run "terraform init"
```

**Solution:**
```bash
terraform init
```

### Error: S3 Bucket Already Exists

```
Error: creating Amazon S3 Bucket: BucketAlreadyExists
```

**Solution:**
1. Use a unique bucket name in `variables.tf`
2. Or import existing bucket: `terraform import aws_s3_bucket.terraform_state bucket-name`

### Error: EKS Cluster Creation Timeout

```
Error: waiting for EKS Cluster creation: timeout
```

**Solution:**
- EKS cluster creation takes 10-15 minutes
- Increase timeout or wait and retry
- Check AWS Console for cluster status

### Error: VPC Limit Exceeded

```
Error: VpcLimitExceeded
```

**Solution:**
```bash
# Check current VPCs
aws ec2 describe-vpcs --query 'Vpcs[*].VpcId'

# Delete unused VPCs or request limit increase
```

---

## EKS Issues

### Cannot Connect to Cluster

```
error: You must be logged in to the server (Unauthorized)
```

**Solution:**
```bash
# Update kubeconfig
aws eks update-kubeconfig --region ap-south-1 --name fintech-compliance

# Verify context
kubectl config current-context
```

### Nodes Not Joining Cluster

```bash
kubectl get nodes
# No resources found
```

**Solution:**
1. Check node group status in AWS Console
2. Verify IAM roles are correct
3. Check security groups allow node-to-control-plane communication

```bash
# Check node group events
aws eks describe-nodegroup \
  --cluster-name fintech-compliance \
  --nodegroup-name fintech-compliance-nodes
```

---

## Kubernetes Issues

### Pod ImagePullBackOff

```bash
kubectl get pods
# STATUS: ImagePullBackOff
```

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Wrong image URL | Verify ECR URL in deployment |
| ECR auth issues | Check node IAM role has ECR permissions |
| Image doesn't exist | Push image to ECR first |

```bash
# Debug
kubectl describe pod <pod-name>
```

### Pod CrashLoopBackOff

```bash
kubectl get pods
# STATUS: CrashLoopBackOff
```

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| App error | Check logs: `kubectl logs <pod>` |
| Missing env vars | Verify ConfigMap/Secret |
| Resource limits | Increase memory/CPU limits |

```bash
# Check previous logs
kubectl logs <pod-name> --previous
```

### Service No External IP

```bash
kubectl get svc
# EXTERNAL-IP: <pending>
```

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Subnet tags missing | Add kubernetes.io/role/elb tag |
| Security group issues | Check SG allows traffic |
| Load balancer limit | Check AWS LB quotas |

```bash
# Debug
kubectl describe svc fintech-simplifier
```

### Pod Pending - Insufficient Resources

```
Warning  FailedScheduling  pod has unbound immediate PersistentVolumeClaims
```

**Solution:**
```bash
# Check node resources
kubectl describe nodes

# Scale down or add nodes
```

---

## ECR Issues

### Push Permission Denied

```
denied: User: arn:aws:iam::xxx is not authorized to perform: ecr:InitiateLayerUpload
```

**Solution:**
```bash
# Re-authenticate
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin <ecr-url>

# Check IAM permissions include ecr:* actions
```

### Image Not Found

```
manifest for xxx not found
```

**Solution:**
```bash
# List images in repository
aws ecr list-images --repository-name fintech-compliance

# Push correct tag
docker push <ecr-url>:latest
```

---

## Application Issues

### 502 Bad Gateway

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Pods not ready | Check pod health: `kubectl get pods` |
| Wrong port | Verify targetPort matches container port |
| App crashed | Check logs: `kubectl logs <pod>` |

### Slow Response Times

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| LLM API slow | NVIDIA API latency is normal |
| Resource limits | Increase CPU/memory |
| Single replica | Scale to more replicas |

```bash
# Check resource usage
kubectl top pods
```

### NVIDIA API Errors

```
Error: Invalid API key
```

**Solution:**
```bash
# Verify secret value
kubectl get secret fintech-secrets -o jsonpath='{.data.NVIDIA_API_KEY}' | base64 -d

# Update secret
kubectl delete secret fintech-secrets
kubectl create secret generic fintech-secrets \
  --from-literal=NVIDIA_API_KEY=nvapi-correct-key

# Restart pods
kubectl rollout restart deployment/fintech-simplifier
```

---

## Network Issues

### DNS Resolution Failure

```
Error: dial tcp: lookup xxx: no such host
```

**Solution:**
```bash
# Check CoreDNS
kubectl get pods -n kube-system -l k8s-app=kube-dns

# Restart CoreDNS if needed
kubectl rollout restart deployment/coredns -n kube-system
```

### Timeout Connecting to External Services

**Solution:**
- Check NAT Gateway is running
- Verify security groups allow outbound traffic
- Check route tables have NAT gateway route

---

## Cleanup Issues

### Cannot Delete VPC

```
Error: DependencyViolation: The vpc has dependencies
```

**Solution:**
```bash
# Delete resources in order:
# 1. Kubernetes resources
kubectl delete -f infrastructure/backend/k8s/

# 2. EKS cluster (via terraform)
# 3. VPC (via terraform)

terraform destroy
```

### Terraform State Lock

```
Error: Error acquiring the state lock
```

**Solution:**
```bash
# Force unlock (use carefully!)
terraform force-unlock <lock-id>
```

---

## Getting Help

### Collect Debug Information

```bash
# Cluster info
kubectl cluster-info dump > cluster-dump.txt

# Pod details
kubectl describe pods > pods-describe.txt

# Events
kubectl get events --sort-by='.lastTimestamp' > events.txt
```

### AWS Support

- Check [AWS Service Health Dashboard](https://status.aws.amazon.com/)
- Open support ticket via AWS Console

### Resources

- [EKS Documentation](https://docs.aws.amazon.com/eks/)
- [Kubernetes Docs](https://kubernetes.io/docs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest)
