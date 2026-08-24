# AWS Infrastructure Details

## EKS Cluster (Kubernetes)
- **Name:** `AHS-EHR-Dev-arcaquest-eks` (k8s 1.33)
- **VPC:** `vpc-08d377400b7d3bf72` (3 private subnets across 1a/1b/1c)
- **API Endpoint:** Private only (kubectl must run from AHS-EHR-Dev-Bastion-EC2 `65.0.32.20`)
- **Node Groups:** (All on-demand, no taints/labels)
  - `Worker-node`: t3.large (Scale: 2/5/6)
  - `cost-optimization-node`: t3.large + c6a.xlarge (Scale: 1/2/4)
  - `superset-node`: t3a.medium + t3.large (Scale: 1/1/4)
  - `Rerankers-node`: g4dn.xlarge GPU (Scale: 1/2/4)
- **Ingress:** Shared ALB via AWS LB Controller, host-based routing

## ArcaQuest Workload
- **Namespace:** `arcaquest-dev`
- **ServiceAccount:** `ehr-app-sa` (IRSA)
- **Host:** `arcaquest-dev.arcaai.com` -> svc port `8000`
- **ECR Repositories:** 
  - `ahs-ehr-dev-arcaquest/arcaquest-ehr`
  - `ahs-ehr-dev-arcaquest/phidata/pgvector`
- **Deployment Strategy:** Manual deploys (kubectl/helm from bastion, no CodePipeline)

## Redis (ElastiCache)
- **Replication Group:** `ahs-ehr-dev-redis-rg` (Redis 7.1.0, cluster mode disabled, single node)
- **Endpoint:** `master.ahs-ehr-dev-redis-rg.whubdm.aps1.cache.amazonaws.com:6379`
- **Node Type:** `cache.t3.micro`
- **Security:** 
  - TLS in-transit is ON (mode preferred, both `redis://` and `rediss://` work)
  - No auth token (network-scoped access only)
  - Located in the same VPC as EKS. SG allows port 6379 (pods can connect with zero network changes)

## Access & Operations
- **Kubeconfig Update (from bastion):**
  ```bash
  aws eks update-kubeconfig --name AHS-EHR-Dev-arcaquest-eks --region ap-south-1
  ```
- **Registered EKS Access Entries:**
  - bastion role
  - `Arcaai-EKS-AdminRole`
  - 2 CodeBuild roles
  - `AWSAdministratorAccess`
- **WARNING:** Plain `PowerUserAccess` SSO is not in the list. An access entry needs to be added before kubectl will authorize.
