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
| `milabench run`     | **Multi-node Slurm allocation**, command launched from the `main` node (= the vLLM server) | milabench asserts the run is launched from the `main` node, then SSHs from there to every node listed in `system.yaml`. |

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

> **⚠️ `~/.bashrc` exports do NOT reach the remote trainer/vLLM ranks.**
> milabench runs every node command over a **non-interactive** `ssh host -- …`
> shell, and the stock `~/.bashrc` bails out at the top on the
> `case $- in *i*) ;; *) return;; esac` guard before any export runs. So the
> vars above are inherited only by the **launch node** (your interactive
> shell). Anything the *remote ranks* need at runtime — most importantly
> `NCCL_SOCKET_IFNAME` — must be injected through milabench's own command
> chain, i.e. `benchmarks/grpo-large/scripts/with_modules` (see Step 5 and
> Troubleshooting). Don't try to fix NCCL by editing `~/.bashrc`; it silently
> does nothing on the remote side.

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

# First node = main = vLLM server = the node you launch milabench from.
# Remaining 4 nodes = trainers.
i=0
for host in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
  ip=$(getent hosts "$host" | awk '{ print $1 }')
  main=$([ $i -eq 0 ] && echo "true" || echo "false")   # first node = main = vLLM
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

**Why the first node is marked `main`.** Three milabench rules chain
together and force a single topology:

1. `milabench run` asserts you launch it **from the `main` node**
   (`milabench/multi.py`: `assert is_main_local(...)`, "Running
   benchmarks only works on the main node"). `self`/`main` is matched by
   the launching machine's own IP.
2. `select_nodes` always moves the `main` node to **index 0** of the
   node list ("main node is always first").
3. `benchfile.py` slices the vLLM server off the front:
   `vllm_nodes = nodes[:vllm_machines]` (index 0).

⇒ **launch node = `main` node = vLLM server** — they are necessarily the
same machine. The remaining four are trainers; trainer rank-0 is the
first non-`main` node and becomes the accelerate rendezvous
(`--main_process_ip`). There is no valid setup where you launch from a
trainer and vLLM lives elsewhere. (The milabench orchestrator is a
lightweight coordinator that uses no GPUs, so it co-exists fine with the
vLLM server's 8-GPU `tensor_parallel_size=8` on that node.)

> **Important:** SSH from your current shell to the `main` (first) node
> and run the rest of the commands there. If you launch from a node not
> in `system.nodes`, the `is_main_local` assertion fails. Easiest:
>
> ```bash
> ssh $(scontrol show hostnames $SLURM_JOB_NODELIST | sed -n '1p')
> ```

---

## Step 5 — Run

On the `main` node (the first node = the vLLM server, which you SSH'd into at the end of step 4):

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

> **Fixes already baked into this repo (learned the hard way):**
> 1. **NCCL interface** — `scripts/with_modules` exports
>    `NCCL_SOCKET_IFNAME=bond0` so NCCL uses the routable `10.28.1.x` net
>    instead of an unroutable IB-over-IP rail. Without it, trainers die
>    ~1–4 min in with `No route to host`.
> 2. **TRL arg** — `--max_prompt_length` was removed from
>    `config/training.yaml`; TRL 1.5.1's `GRPOConfig` no longer accepts it.
> 3. **DeepSpeed GAS** — a top-level `gradient_accumulation_steps: 8` was
>    added so the accelerate/DeepSpeed-plugin value matches the `argv` one;
>    otherwise HF aborts at `deepspeed_init` with a `ds=1 vs hf=8` mismatch.
>
> All three live in the shared `_grpo_large` config / wrapper, so the 122B
> run inherits them too. See Troubleshooting if you hit any of these symptoms
> on a different cluster or TRL pin.

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
| Trainers die ~1–4 min in with `socketPollConnect … No route to host` / `ncclRemoteError: remote process exited prematurely` | NCCL auto-selected an interface that isn't routable between nodes. On this cluster the IB-over-IP rails (`100.126.x.x` / `100.127.x.x`) are **not** IP-routable peer-to-peer, but NCCL picks one anyway. The routable net is `bond0` (`10.28.1.x`) — the same one SSH uses. | **Already handled:** `scripts/with_modules` now exports `NCCL_SOCKET_IFNAME=bond0` for every remote rank. If your cluster's routable iface differs, override it: `export NCCL_SOCKET_IFNAME=<iface>` before the run (the wrapper uses `${NCCL_SOCKET_IFNAME:-bond0}`), or edit the wrapper. **Do not** put it in `~/.bashrc` — non-interactive SSH ignores it (see Prerequisites). Verify routability with `ping <peer-bond0-ip>` vs `ping <peer-ib-ip>`. |
| Hang at first `update_named_param` | NCCL can't reach the trainer↔vLLM peers | Same root cause / fix as the row above — pin `NCCL_SOCKET_IFNAME` to the routable iface (`bond0` here). |
| All ranks die instantly with `ValueError: Some specified arguments are not used by the HfArgumentParser: ['--max_prompt_length', …]` | Config↔library version skew: the installed **TRL 1.5.1** removed `max_prompt_length` from `GRPOConfig` (only `max_completion_length` remains). The earlier `0.18.x` warning in the logs is about **vLLM**, not TRL — don't be misled. | Already fixed: `--max_prompt_length` was removed from the `_grpo_large` block in `config/training.yaml`. If you change the TRL pin, dry-parse the argv first: `python -c "from transformers import HfArgumentParser; from trl import GRPOConfig, ModelConfig; from trl.scripts.utils import ScriptArguments; print(HfArgumentParser((ScriptArguments,GRPOConfig,ModelConfig)).parse_known_args([...])[1])"` and drop whatever it reports as unrecognized. |
| Crash at `trainer.train()` → `deepspeed_init` with `ValueError: … ds gradient_accumulation_steps=1 vs hf gradient_accumulation_steps=8` | milabench builds the `accelerate launch --gradient_accumulation_steps=N` flag (which configures the DeepSpeed plugin) from a **top-level** config key `gradient_accumulation_steps` (default **1**), *separately* from the `--gradient_accumulation_steps` under `argv:` (which sets HF `TrainingArguments`). If only the argv one is set, the two disagree and HF refuses to start. | Already fixed: a top-level `gradient_accumulation_steps: 8` was added to `_grpo_large` to match the argv value. **Keep the two in sync** — any change to the argv `--gradient_accumulation_steps` must be mirrored in the top-level key. |
| `Address already in use` on port 8000, **or** `milabench run` hangs forever after the trainers have already crashed | milabench does **not** kill the persistent vLLM server when trainers crash, so the orchestrator blocks waiting on it. ⚠️ `pkill -f vllm-serve` is **not enough**: vLLM's `tensor_parallel_size=8` spawns 8 worker subprocesses that **rename themselves to `VLLM::Worker_TP0..7`**, so they don't match `vllm-serve` and survive — each still pinning ~76 GB on its GPU. | Full teardown on the vLLM node: kill the workers by PID from nvidia-smi, then the parent: `nvidia-smi --query-compute-apps=pid --format=csv,noheader \| xargs -r kill -9; pkill -9 -f 'trl.*vllm-serve'`. Also kill the orchestrator **by PID** (`pgrep -f 'milabench run'` then `kill -9 <pid>` — don't `pkill -f "milabench run"`, the pattern matches its own command line and kills the wrong shell). On the trainer nodes, sweep leftovers: `pkill -9 -f 'grpo-large/main.py'`. Confirm GPUs are back to ~0 MiB before relaunching. |
| `MissingCUDAException: CUDA_HOME does not exist` (DeepSpeed) | SSH runs a non-login shell that never sources the module system | Handled automatically: the benchfile wraps every remote command with `scripts/with_modules`, loading the modules in the bench's `modules:` config key (default `cuda13.0/toolkit/13.0.2 nccl2-cuda13.0-gcc/2.28.9`). The same wrapper is also the place that exports `NCCL_SOCKET_IFNAME` to the remote ranks (see the NCCL row below). Adjust both for a different cluster. |
| `trl: command not found` on the vLLM node | venv not activated in the SSH shell | Handled automatically: the vLLM command runs through `with_modules … -- activator <venv> <cache> trl …`. If you customize the benchfile, keep the `activator` in the chain. |
| `main.py: error: argument --bf16: expected one argument` | milabench renders YAML `true` as a **bare flag**, but this parser requires a value for `--bf16` | Quote the value in `config/training.yaml`: `--bf16: "True"` (renders as `--bf16 True`). `--gradient_checkpointing: true` is fine bare — its parser entry accepts no-arg. |
| OOM on trainer GPU after a few steps | Activation memory under-budgeted | Drop `--max_completion_length` to 4096 or `--num_generations` to 1 in the YAML; re-run. |
| `ssh: Permission denied (publickey)` | Public key not in `~/.ssh/authorized_keys` (milabench SSHes to every node, including back to itself by IP) | `cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys`. With a shared-NFS `$HOME` this fixes all nodes at once. |
| `AssertionError: Running benchmarks only works on the main node` | You launched `milabench run` from a node that isn't marked `main: true` | Launch from the `main` (first/vLLM) node — `ssh $(scontrol show hostnames $SLURM_JOB_NODELIST \| sed -n '1p')`. See step 4. |
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

# Run (5-node salloc, from the main node = first node = vLLM server)
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
    if [ "$i" -eq 0 ]; then main=true; else main=false; fi   # first node = main = vLLM
    echo "    - name: node$i"
    echo "      hostname: $host"
    echo "      ip: $ip"
    echo "      user: $USER"
    echo "      main: $main"
    i=$((i+1))
  done
} > "$SYSTEM_YAML"

# --- Step 5 (inlined): run on the main node (= node0, vLLM server) ----
MAIN_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '1p')
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
clobber each other) and SSHes from the Slurm batch host onto the `main`
node (node0 = the vLLM server) before invoking `milabench run` — same
topology as the interactive path, satisfying milabench's
"run from the main node" assertion.
