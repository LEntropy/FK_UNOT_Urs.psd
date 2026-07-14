#!/bin/bash
# No root needed -- data/ dir is philosophyz-owned and world-writable;
# global system python3 has the deps installed (no dedicated venv on this
# host). A previous session apparently ran this under sudo unnecessarily.
cd /media/philosophyz/SSD/dontai/detection-svc || exit 1
nohup python3 server.py > /media/philosophyz/SSD/dontai/detection-svc.log 2>&1 &
disown
echo "started, PID $!"
