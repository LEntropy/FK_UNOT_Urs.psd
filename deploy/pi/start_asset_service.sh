#!/bin/bash
cd /media/philosophyz/SSD/dontai/asset-service || exit 1
nohup npx tsx src/index.ts > /media/philosophyz/SSD/dontai/asset-service.log 2>&1 &
disown
echo "started, PID $!"
