#!/bin/bash
export ASSET_SERVICE_URL=http://localhost:3002
export PORT=4000
# 127.0.0.1, not "localhost" -- Node resolves "localhost" to ::1 first on
# this host, and the KMS C server only binds IPv4 (AF_INET), so the
# connection gets refused. Hit this for real the first time blockchain-svc
# needed unwrapKey() at startup.
export KMS_HOST=127.0.0.1
export KMS_PORT=8443

cd /media/philosophyz/SSD/dontai/apps/api-gateway || exit 1
nohup node dist/index.js > /media/philosophyz/SSD/dontai/api-gateway.log 2>&1 &
disown
echo "started, PID $!"
