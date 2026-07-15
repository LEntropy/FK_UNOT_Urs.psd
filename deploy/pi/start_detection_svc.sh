#!/bin/bash
# No root needed -- data/ dir is philosophyz-owned and world-writable.
# Uses .venv/bin/python (not global python3, which doesn't have fastapi
# etc. installed) -- run `python3 -m venv .venv && ./.venv/bin/python -m
# pip install -r requirements.txt` once if .venv doesn't exist yet.
cd /media/philosophyz/SSD/dontai/detection-svc || exit 1
nohup ./.venv/bin/python server.py > /media/philosophyz/SSD/dontai/detection-svc.log 2>&1 &
disown
echo "started, PID $!"
