
# grpo-large

Multi-node GRPO benchmark for large Qwen checkpoints on
`nvidia/AceReason-Math`. Trains the policy with TRL's `GRPOTrainer`,
generates rollouts via co-located vLLM, and rewards correctness with a
rule-based math verifier (`math_verify`) on the dataset's gold `answer`
column. No separate reward model is loaded, so the per-GPU footprint is
just the policy plus its Zero-3 shards.

| Setting | Value |
| --- | --- |
| Trainer | `trl.GRPOTrainer` |
| Reward | `math_verify`-based exact / equivalence check vs gold `answer` |
| Dataset | `nvidia/AceReason-Math` (`train` split, columns `problem`→`prompt`, `answer`) |
| Rollouts | co-located vLLM (`vllm_mode: colocate`) |
| Sharding | DeepSpeed Zero-3 + CPU optimizer/parameter offload via Accelerate |

## Variants

| Bench | Model | `num_machines` (suggested) |
| --- | --- | --- |
| `grpo-large-72b`  | `Qwen/Qwen2.5-72B`         | 1 (8×80GB) |
| `grpo-large-122b` | `Qwen/Qwen3.5-122B-A10B`   | 2          |
| `grpo-large-397b` | `Qwen/Qwen3.5-397B-A17B`   | 8          |

The node counts are first-pass estimates assuming 80GB-class GPUs;
override via `num_machines` in the YAML to match the actual cluster.

## How multi-node is launched

`benchfile.py` returns
`AccelerateZero3AllNodes(PackCommand(...))`. Milabench's
`AccelerateAllNodes` SSHes to each node in `system.nodes` and prepends
`accelerate launch --num_machines/--machine_rank/--main_process_ip`. The
`AccelerateZero3AllNodes` subclass appends Zero-3 + offload flags so the
final command shards parameters across all visible GPUs.

## Local dev

```bash
cd benchmarks/grpo-large
milabench install --config dev.yaml --base .
milabench prepare --config dev.yaml --base .
milabench run     --config dev.yaml --base .
```

`dev.yaml` uses `Qwen/Qwen2.5-1.5B` so the bench is iterable on a
single-GPU box without burning a real allocation.
