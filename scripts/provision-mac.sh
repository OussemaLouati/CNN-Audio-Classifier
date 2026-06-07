#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="audio-ml"
NAMESPACE="audio-classifier"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Deleting old cluster if exists ==="
k3d cluster delete "${CLUSTER_NAME}" 2>/dev/null || true

echo "=== Creating k3d cluster ==="
k3d cluster create "${CLUSTER_NAME}" \
  --port "8080:80@loadbalancer" \
  --port "9000:9000@loadbalancer" \
  --port "5000:5000@loadbalancer" \
  --agents 1

kubectl wait --for=condition=ready node --all --timeout=60s

echo "=== Creating namespace ==="
kubectl create namespace "${NAMESPACE}"
kubectl config set-context --current --namespace="${NAMESPACE}"

echo "=== Installing MinIO ==="
helm repo add minio https://charts.min.io/ 2>/dev/null || true
helm repo update >/dev/null
helm install minio minio/minio \
  --namespace "${NAMESPACE}" \
  --set rootUser=minioadmin \
  --set rootPassword=minioadmin123 \
  --set mode=standalone \
  --set replicas=1 \
  --set persistence.size=10Gi \
  --set resources.requests.memory=512Mi \
  --set service.type=ClusterIP \
  --set consoleService.type=ClusterIP \
  --set 'buckets[0].name=mlflow-artifacts' \
  --set 'buckets[0].policy=none' \
  --set 'buckets[1].name=pipeline-data' \
  --set 'buckets[1].policy=none' \
  --wait --timeout 180s

echo "=== Creating secrets ==="
kubectl create secret generic minio-credentials \
  --namespace "${NAMESPACE}" \
  --from-literal=access-key=minioadmin \
  --from-literal=secret-key=minioadmin123

kubectl create secret generic seldon-rclone-secret \
  --namespace "${NAMESPACE}" \
  --from-literal=RCLONE_CONFIG_S3_TYPE=s3 \
  --from-literal=RCLONE_CONFIG_S3_PROVIDER=Minio \
  --from-literal=RCLONE_CONFIG_S3_ACCESS_KEY_ID=minioadmin \
  --from-literal=RCLONE_CONFIG_S3_SECRET_ACCESS_KEY=minioadmin123 \
  --from-literal=RCLONE_CONFIG_S3_ENDPOINT=http://minio:9000 \
  --from-literal=RCLONE_CONFIG_S3_ENV_AUTH=false

echo "=== Creating configmaps ==="
kubectl create configmap minio-config \
  --namespace "${NAMESPACE}" \
  --from-literal=endpoint=minio:9000 \
  --from-literal=endpoint_url=http://minio:9000 \
  --from-literal=bucket=mlflow-artifacts

kubectl create configmap mlflow-config \
  --namespace "${NAMESPACE}" \
  --from-literal=tracking_uri=http://mlflow:5000

kubectl create configmap pipeline-config \
  --namespace "${NAMESPACE}" \
  --from-file=pipeline_config.yaml="${PROJECT_ROOT}/configs/pipeline_config.yaml"

echo "=== Deploying MLflow ==="
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
        - name: mlflow
          image: ghcr.io/mlflow/mlflow:v2.11.0
          command: ["mlflow", "server"]
          args: ["--host=0.0.0.0", "--port=5000", "--backend-store-uri=sqlite:///mlflow/mlflow.db", "--default-artifact-root=s3://mlflow-artifacts"]
          ports:
            - containerPort: 5000
          env:
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: access-key
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-credentials
                  key: secret-key
            - name: MLFLOW_S3_ENDPOINT_URL
              value: "http://minio:9000"
          volumeMounts:
            - name: data
              mountPath: /mlflow
      volumes:
        - name: data
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: ${NAMESPACE}
spec:
  selector:
    app: mlflow
  ports:
    - port: 5000
      targetPort: 5000
EOF

kubectl wait --for=condition=ready pod -l app=mlflow --timeout=500s -n "${NAMESPACE}"

echo "=== Installing Argo Workflows ==="
helm repo add argo https://argoproj.github.io/argo-helm 2>/dev/null || true
helm repo update >/dev/null
helm install argo-workflows argo/argo-workflows \
  --namespace "${NAMESPACE}" \
  --set server.serviceType=ClusterIP \
  --set 'server.authModes={server}' \
  --set "controller.workflowNamespaces={${NAMESPACE}}" \
  --set workflow.serviceAccount.create=true \
  --set workflow.serviceAccount.name=argo-workflow \
  --set controller.containerRuntimeExecutor=emissary \
  --wait --timeout 120s

echo "=== Setting up RBAC ==="
kubectl create rolebinding argo-workflow-admin \
  --clusterrole=admin \
  --serviceaccount="${NAMESPACE}:argo-workflow" \
  --namespace="${NAMESPACE}"

kubectl create rolebinding default-admin \
  --clusterrole=admin \
  --serviceaccount="${NAMESPACE}:default" \
  --namespace="${NAMESPACE}"



echo "=== Installing Seldon Core ==="
helm repo add seldonio https://storage.googleapis.com/seldon-charts 2>/dev/null || true
helm repo update >/dev/null
docker pull --platform linux/amd64 seldonio/seldon-core-operator:1.17.1 2>/dev/null || true
k3d image import seldonio/seldon-core-operator:1.17.1 -c "${CLUSTER_NAME}" 2>/dev/null || true
helm install seldon-core seldonio/seldon-core-operator \
  --namespace "${NAMESPACE}" \
  --version 1.17.1 \
  --set usageMetrics.enabled=false \
  --set istio.enabled=false \
  --set ambassador.enabled=false \
  --set image.pullPolicy=IfNotPresent \
  --wait --timeout 120s

kubectl create clusterrole seldon-manage \
  --verb=get,list,patch,update,create,delete \
  --resource=seldondeployments.machinelearning.seldon.io 2>/dev/null || true

kubectl create rolebinding argo-seldon-manage \
  --clusterrole=seldon-manage \
  --serviceaccount="${NAMESPACE}:argo-workflow" \
  --namespace="${NAMESPACE}" 2>/dev/null || true
  
echo "=== Creating PVC ==="
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pipeline-data-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
EOF

echo "=== Building pipeline image ==="
docker build -t audio-classifier-pipeline:latest "${PROJECT_ROOT}"
k3d image import audio-classifier-pipeline:latest -c "${CLUSTER_NAME}"

echo "=== Applying Seldon Deployment ==="
kubectl apply -f "${PROJECT_ROOT}/deploy/k8s/manifests/seldon-deployment.yaml"

echo "=== Uploading data to MinIO ==="
kubectl port-forward svc/minio 19000:9000 -n "${NAMESPACE}" &
PF_PID=$!
sleep 3

mc alias set pipeline http://localhost:19000 minioadmin minioadmin123 --quiet 2>/dev/null || true

# Upload any .wav/.txt files in project root
for f in "${PROJECT_ROOT}"/*.wav "${PROJECT_ROOT}"/*.txt; do
  if [ -f "$f" ]; then
    mc cp "$f" pipeline/pipeline-data/input/
  fi
done

mc ls pipeline/pipeline-data/input/ 2>/dev/null || true
kill $PF_PID 2>/dev/null || true

echo ""
echo "=== Done ==="
echo ""
echo "Run the pipeline:"
echo "  argo submit deploy/k8s/argo-workflow.yaml -n ${NAMESPACE}"
echo ""
echo "UIs:"
echo "  MLflow: kubectl port-forward svc/mlflow 5000:5000 -n ${NAMESPACE}"
echo "  Argo:   kubectl port-forward svc/argo-workflows-server 2746:2746 -n ${NAMESPACE}"
echo "  MinIO:  kubectl port-forward svc/minio-console 9001:9001 -n ${NAMESPACE}"
