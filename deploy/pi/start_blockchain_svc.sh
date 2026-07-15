#!/bin/bash
cd /media/philosophyz/SSD/dontai/blockchain-svc || exit 1
nohup npx tsx src/index.ts > /media/philosophyz/SSD/dontai/blockchain-svc.log 2>&1 &
disown
echo "started, PID $!"
