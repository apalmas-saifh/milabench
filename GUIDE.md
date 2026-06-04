# Running `grpo-large-72b` and `grpo-large-122b` from scratch

End-to-end runbook for executing the two large-model GRPO benchmarks
defined in `benchmarks/grpo-large/` against `nvidia/AceReason-Math` with
TRL + DeepSpeed Zero-3 + dedicated vLLM rollout nodes.

Each bench needs **5 nodes** of 8×H100-80GB: 1 vLLM server + 4 trainers.

---

## Where to run each step

| Action              | Where                                                                     | Why                                                                                              |
| ---                 | ---                                                                       | ---                                                                                              |
| Clone + venv        | Login node OK                                                             | No GPU work, just file operations.                                                               |
| `milabench pin`     | Login node OK (you've already done this)                                  | Pure pip resolver; no GPU.                                                                       |
| `milabench install` | **Compute node** (1 GPU enough)                                           | `vllm` / `deepspeed` / `torch` wheels probe `nvcc` / `nvidia-smi`. Login node will OOM the resolver and may kill long pip builds. |
| `milabench prepare` | **Compute node** (1 GPU) **or** a network/data-staging node              | Downloads 145–244 GB from HuggingFace. Don't blow up login-node disk quotas.                     |
| `milabench run`     | **Multi-node Slurm allocation**, command launched from the rank-0 trainer | The bench SSHs from there to every other node listed in `system.yaml`.                           |

`install` + `prepare` can share a single 1-GPU `salloc`. `run` needs
the full 5-node allocation.

---

## Prerequisites (one-time)

```bash
# On the login node
cd /lambdafs/users/a.palmas
git clone https://github.com/mila-iqia/milabench.git    # or your fork
cd milabench

# uv-managed venv (matches the workflow you've been using)
uv sync

# Persistent env vars — put these in ~/.bashrc so SSH non-login shells inherit them
cat >> ~/.bashrc <<'EOF'
export MILABENCH_BASE=/lambdafs/users/a.palmas/milabench
export MILABENCH_CONFIG=$MILABENCH_BASE/config/standard.yaml
export MILABENCH_GPU_ARCH=cuda
EOF
source ~/.bashrc

# Hugging Face token for downloading models (Qwen weights are public but the
# rate limits are unauthenticated-user-grade; a token lifts them).
export MILABENCH_HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Passwordless SSH between cluster nodes** must be working:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ''
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
# Verify: from any allocated node you should be able to do
ssh <other_allocated_node> hostname
# ... without a password prompt.
```

---

## Step 1 — Pin requirements (login node OK)

Already done on your side, but for reference:

```bash
cd milabench
uv run milabench pin --select grpo-large-72b --variant cuda
# Same .cuda.txt covers grpo-large-122b — they share a venv (install_group: torch).
```

This produces `benchmarks/grpo-large/requirements.cuda.txt`. Commit it.

---

## Step 2 — Install + prepare (1-GPU compute node)

Grab a single GPU for a few hours; everything below runs inside this
allocation.

```bash
salloc --nodes=1 --gpus-per-task=h100:1 --cpus-per-task=16 --time=6:00:00 --exclusive
# You're now on a compute node.

module load cuda13.0/toolkit/13.0.2 nccl2-cuda13.0-gcc/2.28.9
cd $MILABENCH_BASE       # repo root
# (env vars already exported via ~/.bashrc)

uv run milabench install --select grpo-large-72b
# This builds the venv at $MILABENCH_BASE/venv/torch and installs every
# pin from requirements.cuda.txt. Expect ~10–20 min.

uv run milabench prepare --select grpo-large-72b
# Downloads Qwen/Qwen2.5-72B (~145 GB) + nvidia/AceReason-Math.
# Goes under $MILABENCH_BASE/cache (HF_HOME is set by milabench). Expect
# 30–60 min on a fast link.

# Repeat prepare for the 122B model (shares the same venv):
uv run milabench prepare --select grpo-large-122b
# Downloads Qwen/Qwen3.5-122B-A10B (~244 GB). Same place, same scheme.

exit   # release the single-GPU allocation
```

After this you have everything cached locally; the multi-node run won't
need network for weights or dataset.

---

## Step 3 — Allocate 5 nodes for the actual run

Use interactive `salloc` for the first run so you can watch the output
of every step; switch to `sbatch` later for repeats (template in the
appendix).

```bash
salloc \
  --nodes=5 \
  --ntasks-per-node=1 \
  --gpus-per-task=h100:8 \
  --cpus-per-task=128 \
  --mem=0 \
  --time=4:00:00 \
  --exclusive

# Slurm puts you on the first allocated node. List the hosts:
scontrol show hostnames $SLURM_JOB_NODELIST
# Example output:
# cn-a001
# cn-a002
# cn-a003
# cn-a004
# cn-a005
```

---

## Step 4 — Write `system.yaml`

Milabench needs a YAML telling it which 5 hosts you got and which one is
the orchestrator. Generate it from the Slurm node list:

```bash
cd $MILABENCH_BASE
mkdir -p runs

cat > runs/system.yaml <<EOF
system:
  arch: cuda
  sshkey: ~/.ssh/id_ed25519
  nodes:
EOF

# First node = vLLM server (rank 0 of the vLLM slice).
# Remaining 4 nodes = trainers (rank 0 of trainers is the orchestrator).
i=0
for host in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
  ip=$(getent hosts "$host" | awk '{ print $1 }')
  main=$([ $i -eq 1 ] && echo "true" || echo "false")   # trainer rank-0 is main
  cat >> runs/system.yaml <<EOF
    - name: node$i
      hostname: $host
      ip: $ip
      user: $USER
      main: $main
EOF
  i=$((i+1))
done

cat runs/system.yaml
```

The first node in the list becomes the vLLM server (`benchfile.py`
slices `vllm_nodes = nodes[:vllm_machines]`). The trainer's rank-0 is
the **second** node — and that's the one that must be marked
`main: true`, because milabench launches the orchestration command from
the `main` node.

> **Important:** SSH from your current shell to the `main` node and run
> the rest of the commands there. milabench's `ForeachNode` SSHes from
> wherever you launch it to each entry in `system.nodes`; if you start
> from a node not in the list, you'll add an extra hop and some
> port-forwarding pain. Easiest:
>
> ```bash
> ssh $(scontrol show hostnames $SLURM_JOB_NODELIST | sed -n '2p')
> ```

---

## Step 5 — Run

On the main (trainer rank-0) node:

```bash
module load cuda13.0/toolkit/13.0.2 nccl2-cuda13.0-gcc/2.28.9
cd $MILABENCH_BASE
# (env vars already exported via ~/.bashrc)

# 72B run
uv run milabench run \
  --system runs/system.yaml \
  --select grpo-large-72b \
  --run-name grpo-large-72b-$(date +%Y%m%d-%H%M%S)

# 122B run (after the 72B finishes; same allocation can serve both)
uv run milabench run \
  --system runs/system.yaml \
  --select grpo-large-122b \
  --run-name grpo-large-122b-$(date +%Y%m%d-%H%M%S)
```

What happens under the hood (from `benchmarks/grpo-large/benchfile.py`):

1. milabench SSHes to **node0** and starts `trl vllm-serve
   --model Qwen/Qwen2.5-72B --tensor_parallel_size 8 --port 8000`.
2. In parallel, milabench SSHes to **node1..node4** and starts
   `accelerate launch ... main.py --use_vllm=True --vllm_mode=server
   --vllm_server_host=<node0_ip> --vllm_server_port=8000` with
   `--num_machines=4 --machine_rank=<r>` and Zero-3 / no-offload flags
   on each.
3. TRL's `VLLMClient` polls `http://<node0_ip>:8000/health` until it
   answers (timeout 1800 s). Then training proceeds: trainer sends
   prompts → vLLM returns completions → trainer computes math-verify
   rewards → policy update → weights pushed back to vLLM via the
   `update_named_param` endpoint.

Outputs land in `$MILABENCH_BASE/runs/<run-name>/`:

```
$MILABENCH_BASE/runs/grpo-large-72b-20260525-143000/
├── grpo-large-72b.0.stdout   # trainer rank-0 (only one with metrics)
├── grpo-large-72b.0.data     # structured metrics
├── grpo-large-72b.1.stdout   # trainer rank-1 (tagged nolog)
├── grpo-large-72b.2.stdout   # trainer rank-2 (tagged nolog)
├── grpo-large-72b.3.stdout   # trainer rank-3 (tagged nolog)
└── grpo-large-72b.vllm.stdout # vLLM server (tagged nolog)
```

Then generate the report:

```bash
uv run milabench report --runs $MILABENCH_BASE/runs/grpo-large-72b-20260525-143000
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `VLLMClient` times out after 1800 s | vLLM server didn't bind, or NCCL port firewalled | Tail `runs/<name>/grpo-large-*.vllm.stdout`. Common: TP=8 needs all 8 GPUs visible on node0; check `CUDA_VISIBLE_DEVICES`. |
| Hang at first `update_named_param` | NCCL can't reach the trainer↔vLLM peers | Set `NCCL_SOCKET_IFNAME=<your IB iface>` in `~/.bashrc`. Default `lo` will hang. |
| `Address already in use` on port 8000 | Another user / leftover process on node0 | Edit `config/training.yaml`'s `_grpo_large.vllm_port` to a random port (e.g. `$RANDOM + 30000`). |
| OOM on trainer GPU after a few steps | Activation memory under-budgeted | Drop `--max_completion_length` to 4096 or `--num_generations` to 1 in the YAML; re-run. |
| `ssh: Permission denied (publickey)` | Cluster doesn't propagate keys to compute nodes | Verify `~/.ssh/authorized_keys` is on the shared FS and readable from compute. |
| Bench skipped with `requires_capabilities` failed | Fewer than 5 nodes in `system.yaml` | Check `scontrol show hostnames`; if Slurm gave you fewer than requested, your queue is constrained. |

---

## Quick reference

```bash
# Pin (login node, one-time per requirements change)
uv run milabench pin --select grpo-large-72b --variant cuda

# Install + prepare (1-GPU salloc)
uv run milabench install --select grpo-large-72b
uv run milabench prepare --select grpo-large-72b
uv run milabench prepare --select grpo-large-122b

# Run (5-node salloc, from trainer rank-0 / main)
uv run milabench run --system runs/system.yaml --select grpo-large-72b
uv run milabench run --system runs/system.yaml --select grpo-large-122b

# Report
uv run milabench report --runs $MILABENCH_BASE/runs/<run-name>
```

---

## Appendix — `sbatch` template

Once the interactive run is green, you can submit the same job
non-interactively. This script inlines steps 4 and 5 (`system.yaml`
generation + `milabench run`), so it stands alone — no helper scripts
needed.

Save as `run-grpo-large.sbatch` next to the repo root:

```bash
#!/bin/bash
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=h100:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --output=grpo-large-%j.out

set -euo pipefail

BENCH="${1:?usage: sbatch run-grpo-large.sbatch <72b|122b>}"

module load cuda13.0/toolkit/13.0.2 nccl2-cuda13.0-gcc/2.28.9
cd "$MILABENCH_BASE"

# --- Step 4 (inlined): write system.yaml from $SLURM_JOB_NODELIST -----
SYSTEM_YAML="runs/system-$SLURM_JOB_ID.yaml"
mkdir -p runs
{
  echo "system:"
  echo "  arch: cuda"
  echo "  sshkey: $HOME/.ssh/id_ed25519"
  echo "  nodes:"
  i=0
  for host in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    ip=$(getent hosts "$host" | awk '{ print $1 }')
    if [ "$i" -eq 1 ]; then main=true; else main=false; fi
    echo "    - name: node$i"
    echo "      hostname: $host"
    echo "      ip: $ip"
    echo "      user: $USER"
    echo "      main: $main"
    i=$((i+1))
  done
} > "$SYSTEM_YAML"

# --- Step 5 (inlined): run on the trainer rank-0 (= node1, main: true) -
MAIN_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '2p')
RUN_NAME="grpo-large-$BENCH-$SLURM_JOB_ID"

ssh -o StrictHostKeyChecking=no "$MAIN_HOST" bash -lc "'
  cd $MILABENCH_BASE &&
  uv run milabench run \
    --system $MILABENCH_BASE/$SYSTEM_YAML \
    --select grpo-large-$BENCH \
    --run-name $RUN_NAME
'"
```

Submit with:

```bash
sbatch run-grpo-large.sbatch 72b
# or
sbatch run-grpo-large.sbatch 122b
```

The script writes its own `system.yaml` under
`$MILABENCH_BASE/runs/system-<jobid>.yaml` (so concurrent jobs don't
clobber each other) and SSHes from the Slurm batch host onto the main
trainer node before invoking `milabench run` — same topology as the
interactive path.
