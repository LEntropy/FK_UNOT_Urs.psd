#!/bin/bash
export ASSET_SERVICE_URL=http://localhost:3002
export PORT=4000
export KMS_HOST=localhost
export KMS_PORT=8443

cd /media/philosophyz/SSD/dontai/apps/api-gateway || exit 1
nohup node dist/index.js > /media/philosophyz/SSD/dontai/api-gateway.log 2>&1 &
disown
echo "started, PID $!"
