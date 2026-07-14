#!/bin/bash
# Starts the existing C KMS server (envelope-key decrypt oracle) on the Pi.
# Must run with cwd = the kms install dir -- kms-server reads keys/, cert/,
# and policy.conf via relative paths (see src/keymgr.c, src/tls.c, src/policy.c).
KMS_DIR="/media/philosophyz/SSD/opt/kms"

cd "$KMS_DIR" || exit 1
nohup ./kms-server > /media/philosophyz/SSD/dontai/kms.log 2>&1 &
disown
echo "started, PID $!"
