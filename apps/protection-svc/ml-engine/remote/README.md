# Remote GPU execution

Runs `style_cloak.py` on a second Windows PC on the same LAN that has an
NVIDIA GPU, instead of the slow CPU path on the dev machine. One-time setup,
then a single script per run.

## One-time setup

1. On the **GPU PC**: follow [SETUP_GPU_PC.md](SETUP_GPU_PC.md) (enable OpenSSH
   Server, note its IP address and username, install Python).
2. On the **GPU PC**: add the dev PC's public key to
   `%USERPROFILE%\.ssh\authorized_keys` (the key is printed at the end of
   SETUP_GPU_PC.md's instructions, or re-print it here with
   `cat ~/.ssh/dontai_gpu_pc.pub`).
3. On the **dev PC** (here): copy `remote.env.example` to `remote.env` and
   fill in `GPU_HOST` (the IP from step 1) and `GPU_USER`.
4. Test the connection:
   ```bash
   source remote.env
   ssh -i "$SSH_KEY" "$GPU_USER@$GPU_HOST" "echo connected"
   ```

## Running a cloak job on the GPU

```bash
./run_remote.sh L3_ANTI_TRAIN cloaked_gpu.png
```

This will (see `run_remote.sh` for the exact steps):
1. Copy `src/`, `scripts/`, `requirements.txt`, and the input images in `out/`
   to the GPU PC.
2. Create a venv there and install CUDA-build PyTorch (`cu121` wheel index —
   if `nvidia-smi` on the GPU PC reports an older driver, edit the index URL
   in `run_remote.sh` to `cu118` instead). This step is slow the *first* time
   only (~2-3GB download); later runs skip it since the venv already exists.
3. Run `src/style_cloak.py` on the GPU.
4. Copy the result back to this machine's `out/` folder.

## Why not rsync

Both machines are Windows, so this uses `scp`/`ssh` (Windows' built-in
OpenSSH client) instead of rsync, which isn't installed on either side by
default. Fine at this scale (a handful of small files); revisit if the
synced file set grows a lot.

## Known limitation

This is a manual, synchronous script — it blocks the dev PC's terminal while
the GPU PC works. Fine for a PoC; if protection-svc later needs real job
queuing (matching the "don't block the upload response" note in
`apps/blockchain-svc/INTEGRATION.md`), replace this with a proper job queue
(Redis/BullMQ per `PROJECT_DESIGN.md`'s stack) rather than scripted SSH.
