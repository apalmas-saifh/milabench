
# grpo-large

Multi-node GRPO benchmark for large Qwen checkpoints on
`nvidia/AceReason-Math`. Trains the policy with TRL's `GRPOTrainer`,
serves rollouts from **dedicated vLLM nodes** (not co-located), and
rewards correctness with a rule-based math verifier (`math_verify`) on
the dataset's gold `answer` column. No separate reward model is loaded.

## Architecture (disaggregated)

```
                       system.nodes
   ┌─────────────────────────────────────────────────────────────┐
   │                                                             │
   │  vllm[0..M-1]                trainer[0..N-1]                │
   │  ┌────────────┐              ┌────────────┐                 │
   │  │ trl vllm-  │   /generate  │ accelerate │  DeepSpeed-     │
   │  │ serve      │ ◀──────────  │ launch     │  Zero3 sharded  │
   │  │ TP=8       │              │ main.py    │  across all     │
   │  │ host the   │  weight push │            │  trainer GPUs   │
   │  │ policy     │ ──────────▶  │            │                 │
   │  └────────────┘              └────────────┘                 │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘
```

The first `vllm_machines` entries of `system.nodes` are SSHed to run
`trl vllm-serve` (TRL's `vllm serve` wrapper that adds the
`update_named_param` endpoint GRPOTrainer needs). The remaining nodes
SSH-launch `accelerate launch ... main.py --vllm_mode=server
--vllm_server_host=<vllm-rank-0-ip> --vllm_server_port=8000`. TRL's
`VLLMClient` polls `/health` until `vllm_server_timeout` elapses, so we
don't need our own readiness wrapper.

## Variants

| Bench              | Model                      | Nodes (vLLM + trainer) | `num_generations` | `max_completion_length` |
| ---                | ---                        | ---:                   | ---:              | ---:                    |
| `grpo-large-72b`   | `Qwen/Qwen2.5-72B`         | 1 + 4 = **5**          | 2                 | 8192                    |
| `grpo-large-122b`  | `Qwen/Qwen3.5-122B-A10B`   | 1 + 4 = **5**          | 4                 | 8192                    |

## Memory budget (Qwen2.5-72B on 4×8 H100-80GB trainer + 1 vLLM node)

| Per-GPU on trainer side | GB |
| --- | ---: |
| Weights (144 GB / 32) | 4.5 |
| Gradients | 4.5 |
| AdamW state (864 GB / 32) | 27 |
| **Static** | **36** |
| Activations @ 8K, batch=1, num_gen=2, grad ckpt | ~25 |
| **Peak per GPU** | **~61** |

Comfortable fit, no offload needed. KL reference policy is disabled
(`--beta 0`) — it would add another 4.5 GB/GPU and isn't needed for a
throughput benchmark. Re-enable by setting `--beta 0.04` (default).

## Local dev

```bash
cd benchmarks/grpo-large
milabench install --config dev.yaml --base .
milabench prepare --config dev.yaml --base .
milabench run     --config dev.yaml --base .
```

`dev.yaml` uses `Qwen/Qwen2.5-1.5B` with 1 vLLM + 1 trainer (still
exercises the disaggregated path end-to-end). You need a `system.yaml`
with at least 2 nodes accessible via passwordless SSH; for single-host
dev you can use loopback by listing the same host twice with different
`name`s (but you'll need to pick different ports for vLLM).
