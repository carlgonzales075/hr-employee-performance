# #!/bin/sh
# sleep 10  # Give MinIO time to start

# mc alias set local http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"
# mc mb local/mlflow || true

#!/bin/sh
sleep 10  # Give MinIO time to start

mc alias set local http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

# Check if the bucket exists before creating it
if ! mc ls local/mlflow >/dev/null 2>&1; then
    mc mb local/mlflow
    echo "Bucket 'mlflow' created."
else
    echo "Bucket 'mlflow' already exists."
fi
