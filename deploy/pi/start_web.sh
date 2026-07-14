#!/bin/bash
export PORT=5173

cd /media/philosophyz/SSD/dontai/web || exit 1
nohup node server.mjs > /media/philosophyz/SSD/dontai/web.log 2>&1 &
disown
echo "started, PID $!"
