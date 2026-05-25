#!/usr/bin/env python

import os
import re
import shutil

# Per-GPU launches (milabench's `per_gpu` plan) all share the same host
# and default to MASTER_PORT=29500. With vllm_mode=colocate, vllm calls
# torch.distributed.init_process_group inside each process, causing
# EADDRINUSE. Derive a unique port from the first visible GPU id.
def _setup_distributed_env():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES") or ""
    first = visible.split(",")[0].strip() if visible else ""
    try:
        offset = int(first)
    except ValueError:
        offset = 0
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + offset))

_setup_distributed_env()

import torch
import accelerate
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
import torchcompat.core as compat

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False


SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize() + ': ' + message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

# Pull a `\boxed{...}` answer if the model emitted one; fall back to the
# last non-empty line.
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return line.rstrip(". ")
    return ""


def math_verify_reward(prompts, completions, answer=None, **_):
    """Rule-based reward on AceReason-Math: 1.0 iff the rollout's final
    answer is mathematically equivalent to the gold `answer` column.

    `GRPOTrainer` passes every extra dataset column as a kwarg here, so as
    long as the train dataset keeps `answer`, this function receives it.
    """
    rewards = []
    golds = answer if answer is not None else [""] * len(completions)
    for completion, gold in zip(completions, golds):
        # `completion` is a string for text-only datasets, or a list of
        # chat-style dicts when the trainer is in conversational mode.
        if isinstance(completion, list):
            completion = completion[-1].get("content", "") if completion else ""
        pred_text = _extract_answer(completion)
        gold_text = str(gold).strip()
        if not pred_text or not gold_text:
            rewards.append(0.0)
            continue
        if _HAS_MATH_VERIFY:
            try:
                pred = mv_parse(f"${pred_text}$")
                gold_p = mv_parse(f"${gold_text}$")
                rewards.append(1.0 if mv_verify(gold_p, pred) else 0.0)
                continue
            except Exception:
                pass
        rewards.append(1.0 if pred_text.replace(" ", "") == gold_text.replace(" ", "") else 0.0)
    return rewards


class GRPOTrainerInstrumented(GRPOTrainer):
    def __init__(self, args: GRPOConfig, *pargs, **kwargs):
        args.report_to = []

        # Same monkeypatch as the rlhf/PPO bench and the small-model grpo
        # bench: swap in milabench's compat Accelerator before the trainer
        # constructs its own.
        accelerate.Accelerator = compat.accelerate.Accelerator
        super().__init__(args=args, *pargs, **kwargs)

        from benchmate.observer import BenchObserver

        max_completion_length = args.max_completion_length

        def batch_size_fn(batch):
            if isinstance(batch, list):
                return len(batch) * max_completion_length
            if isinstance(batch, dict):
                if "input_ids" in batch:
                    shape = batch["input_ids"].shape
                    return shape[0] * shape[-1]
                if "prompt" in batch:
                    return len(batch["prompt"]) * max_completion_length
            return 1

        self._bench_observer = BenchObserver(
            batch_size_fn=batch_size_fn,
            earlystop=70,
            raise_stop_program=True,
            stdout=True,
        )

    def get_train_dataloader(self):
        return self._bench_observer.iterate(super().get_train_dataloader())

    def _save_checkpoint(self, *args, **kwargs):
        pass

    def save_model(self, *args, **kwargs):
        pass


def main():
    from trl.scripts.utils import ScriptArguments

    parser = HfArgumentParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    peft_config = get_peft_config(model_args)

    # AceReason-Math has columns `problem` (the question) and `answer`
    # (the gold short-form answer). GRPO consumes a `prompt` column; we
    # keep `answer` so the math_verify reward fn receives it as a kwarg.
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    if "problem" in dataset.column_names and "prompt" not in dataset.column_names:
        dataset = dataset.rename_column("problem", "prompt")

    eval_samples = min(100, max(1, len(dataset) // 100))
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

    keep_cols = ["prompt"]
    if "answer" in dataset.column_names:
        keep_cols.append("answer")

    with PartialState().local_main_process_first():
        train_dataset = train_dataset.select_columns(keep_cols)
        eval_dataset = eval_dataset.select_columns(keep_cols)

    trainer = GRPOTrainerInstrumented(
        args=training_args,
        model=model_args.model_name_or_path,
        reward_funcs=[math_verify_reward],
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    from voir.phase import StopProgram
    from benchmate.monitor import bench_monitor

    try:
        with bench_monitor():
            main()
    except StopProgram:
        pass
