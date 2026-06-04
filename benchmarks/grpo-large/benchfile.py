import os
from copy import deepcopy

from milabench.pack import Package
from milabench.commands import (
    AccelerateLaunchCommand,
    CmdCommand,
    DockerRunCommand,
    ForeachNode,
    ListCommand,
    PackCommand,
    SSHCommand,
    WrapperCommand,
    activator_script,
    clone_with,
    node_address,
)
from milabench.system import DockerConfig
from milabench.utils import select_nodes

# Default cluster modules to load on each remote node before running. milabench
# SSHes in with a non-login shell that doesn't source the module system, so
# without this DeepSpeed can't find CUDA_HOME. Override via the bench's
# `modules:` config key.
DEFAULT_NODE_MODULES = [
    "cuda13.0/toolkit/13.0.2",
    "nccl2-cuda13.0-gcc/2.28.9",
]


def _with_modules_script():
    """Absolute path to the helper that loads modules then execs a command."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "with_modules")


def _resolve_model_name(argv):
    """Pull `--model_name_or_path <X>` (or `--model_name_or_path=<X>`) out of
    the pack's already-resolved argv list."""
    for i, tok in enumerate(argv):
        s = str(tok)
        if s == "--model_name_or_path" and i + 1 < len(argv):
            return str(argv[i + 1])
        if s.startswith("--model_name_or_path="):
            return s.split("=", 1)[1]
    raise RuntimeError(
        "grpo-large: could not find --model_name_or_path in pack argv; "
        "every grpo-large variant must set it in YAML."
    )


class GRPODisaggregatedNodes(ListCommand):
    """Multi-node GRPO with vLLM rollouts on dedicated nodes.

    Splits `system.nodes` into two groups:
    - the first `vllm_machines` nodes each run `trl vllm-serve` hosting a
      tensor-parallel copy of the policy for fast rollouts;
    - the remaining nodes run `accelerate launch ... main.py` under
      DeepSpeed Zero-3, sharing gradients/optimizer state across all
      trainer GPUs and pulling completions from the vLLM servers via the
      `vllm_mode=server` path that TRL's `VLLMClient` exposes.

    The trainer's `--vllm_server_host` is injected at executor-build time
    so the YAML doesn't need to know which IP rank-0-of-vLLM will land
    on. `VLLMClient` polls `/health` until `--vllm_server_timeout` elapses,
    so we don't need a separate readiness wait wrapper.
    """

    def __init__(
        self,
        executor,
        *extra_accelerate_argv,
        vllm_machines: int = 1,
        vllm_port: int = 8000,
        vllm_tensor_parallel_size: int = 8,
        vllm_server_timeout: int = 1800,
        modules=None,
        **kwargs,
    ) -> None:
        super().__init__(None, **kwargs)
        self.options.update(kwargs)
        self.executor = executor
        self.base_tags = self.executor.pack.config["tag"]
        self.extra_accelerate_argv = extra_accelerate_argv
        self.vllm_machines = vllm_machines
        self.vllm_port = vllm_port
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_server_timeout = vllm_server_timeout
        self.modules = list(modules) if modules else []
        # Run options (e.g. use_stdout) set via set_run_options()/use_stdout()
        # before `executors` is evaluated. We can't forward them to the leaf
        # commands at call time because those are built lazily/fresh inside the
        # `executors` property, so stash them and apply during the build.
        self._run_options = {}

    def set_run_options(self, **kwargs):
        # ListCommand.set_run_options iterates self._executors, but this class
        # builds its executors lazily in the `executors` property (self._executors
        # is the placeholder (None,) from super().__init__). Stash instead and
        # apply to each leaf command we build.
        self._run_options.update(kwargs)
        return self

    def copy(self, pack):
        # ListCommand.copy iterates self._executors (the (None,) placeholder
        # from super().__init__), which would crash on None._set_pack. Like
        # ForeachNode, retarget the template `self.executor` instead; the
        # `executors` property rebuilds the per-node commands from it.
        copy = deepcopy(self)
        copy.executor._set_pack(pack)
        return copy

    def _new_pack(
        self,
        role: str,
        node,
        num_machines,
        has_logs: bool,
        nodes_override=None,
    ):
        config = self.executor.pack.config
        tags = [*self.base_tags, role, node["name"]]
        if not has_logs:
            tags.append("nolog")
        overrides = {"tag": tags}
        if num_machines is not None:
            overrides["num_machines"] = num_machines
        if nodes_override is not None:
            # AccelerateLaunchCommand re-derives `--main_process_ip` and
            # `--num_machines` from `system.nodes`. For the trainer pack
            # we hide the vLLM-role nodes so it only sees the trainer
            # slice.
            overrides["system"] = {"nodes": nodes_override}
        run = clone_with(config, overrides)
        return self.executor.pack.copy(run)

    def _maybe_docker(self, cmd, config):
        docker = config["system"].get("docker")
        if docker:
            return DockerRunCommand(cmd, DockerConfig(**docker))
        return cmd

    def _module_prefix(self):
        # Tokens that load the cluster modules then exec the rest. Always
        # emitted (even with no modules) so the command runs through the same
        # `with_modules ... -- <cmd>` shape; the script is a no-op load when
        # the module list is empty.
        return [_with_modules_script(), *self.modules, "--"]

    def _vllm_executor(self, node, model, key, config):
        pack = self._new_pack(
            role="vllm", node=node, num_machines=None, has_logs=False
        )
        # trl-vllm-serve is the TRL-shipped CLI that wraps `vllm serve`
        # with the extra `update_named_param` endpoint GRPOTrainer needs
        # for weight broadcasts after each policy step.
        #
        # Use CmdCommand (not SimpleCommand): SimpleCommand prepends the
        # pack's full training argv (--output_dir, --dataset_name, ...) to
        # the command, which would be passed to `trl vllm-serve` and, worse,
        # leak in front of the remote command in the SSH argv. CmdCommand
        # runs exactly the tokens we give it.
        #
        # `with_modules ... -- activator <venv> <cache> trl vllm-serve ...`:
        # load cluster modules (CUDA), then the activator activates the venv so
        # `trl` is on PATH (the SSH shell has neither otherwise).
        cmd = CmdCommand(
            pack,
            *self._module_prefix(),
            activator_script(), str(pack.dirs.venv), str(pack.dirs.cache),
            "trl", "vllm-serve",
            "--model", model,
            "--tensor_parallel_size", str(self.vllm_tensor_parallel_size),
            "--host", "0.0.0.0",
            "--port", str(self.vllm_port),
        )
        cmd.set_run_options(**self._run_options)
        cmd = self._maybe_docker(cmd, config)
        return SSHCommand(
            host=node_address(node),
            user=node["user"],
            key=key,
            port=node.get("sshport", 22),
            executor=cmd,
        )

    def _trainer_executor(
        self, rank, node, trainer_nodes, vllm_host, key, config
    ):
        num_trainer = len(trainer_nodes)
        pack = self._new_pack(
            role="trainer",
            node=node,
            num_machines=num_trainer,
            has_logs=(rank == 0),
            nodes_override=trainer_nodes,
        )
        # Reuse the pack's resolved argv (template variables already
        # substituted) and append runtime-only flags pointing at the
        # vLLM rank-0 server.
        base_argv = list(self.executor.pack.argv)
        extra_argv = [
            "--use_vllm=True",
            "--vllm_mode=server",
            f"--vllm_server_host={vllm_host}",
            f"--vllm_server_port={self.vllm_port}",
            f"--vllm_server_timeout={self.vllm_server_timeout}",
        ]
        pack_cmd = PackCommand(pack, *base_argv, *extra_argv, lazy=True)
        # Forward run options (use_stdout) to the leaf pack so trainer rank-0's
        # metrics are scraped from stdout. They bubble up through the command
        # chain's `options` property to execute().
        pack_cmd.set_run_options(**self._run_options)
        acc_cmd = AccelerateLaunchCommand(
            pack_cmd, rank, *self.extra_accelerate_argv
        )
        # Prepend module loading so DeepSpeed finds CUDA_HOME on the remote
        # node. AccelerateLaunchCommand already emits the venv `activator`; the
        # wrapper just loads modules first, i.e. the remote command becomes
        # `with_modules ... -- activator <venv> <cache> accelerate launch ...`.
        acc_cmd = WrapperCommand(acc_cmd, *self._module_prefix())
        acc_cmd = self._maybe_docker(acc_cmd, config)
        return SSHCommand(
            host=node_address(node),
            user=node["user"],
            key=key,
            port=node.get("sshport", 22),
            executor=acc_cmd,
            setsid=(rank == 0),
        )

    @property
    def executors(self):
        config = self.executor.pack.config
        max_num = config.get("num_machines", 1)
        nodes = select_nodes(config["system"]["nodes"], max_num)
        key = config["system"].get("sshkey")

        if len(nodes) < self.vllm_machines + 1:
            raise RuntimeError(
                f"grpo-large disaggregated mode needs at least "
                f"vllm_machines+1 nodes; got {len(nodes)} with "
                f"vllm_machines={self.vllm_machines}."
            )

        vllm_nodes = nodes[: self.vllm_machines]
        trainer_nodes = nodes[self.vllm_machines :]
        vllm_host = node_address(vllm_nodes[0])
        model = _resolve_model_name(list(self.executor.pack.argv))

        cmds = []
        for vn in vllm_nodes:
            cmds.append(self._vllm_executor(vn, model, key, config))
        for r, tn in enumerate(trainer_nodes):
            cmds.append(
                self._trainer_executor(
                    r, tn, trainer_nodes, vllm_host, key, config
                )
            )
        return cmds


class GrpoLarge(Package):
    base_requirements = "requirements.in"
    prepare_script = "prepare.py"
    main_script = "main.py"

    def make_env(self):
        return super().make_env()

    async def install(self):
        await super().install()

    async def prepare(self):
        await super().prepare()

    def build_run_plan(self):
        plan = PackCommand(self, *self.argv, lazy=True)
        # `--zero_stage=3 --zero3_init_flag=true` overrides milabench's
        # hard-coded `--zero_stage=2` (use_deepspeed=true triggers that)
        # by virtue of argparse's "last value wins". No CPU/NVMe offload
        # is configured — the trainer side must fit purely in GPU VRAM.
        vllm_machines = self.config.get("vllm_machines", 1)
        vllm_port = self.config.get("vllm_port", 8000)
        vllm_tp = self.config.get("vllm_tensor_parallel_size", 8)
        vllm_timeout = self.config.get("vllm_server_timeout", 1800)
        # Cluster modules to load on each remote node (CUDA toolkit, NCCL).
        # milabench's SSH shell is non-login and doesn't source the module
        # system, so DeepSpeed otherwise fails with MissingCUDAException.
        modules = self.config.get("modules", DEFAULT_NODE_MODULES)
        return GRPODisaggregatedNodes(
            plan,
            "--zero_stage=3",
            "--zero3_init_flag=true",
            vllm_machines=vllm_machines,
            vllm_port=vllm_port,
            vllm_tensor_parallel_size=vllm_tp,
            vllm_server_timeout=vllm_timeout,
            modules=modules,
        ).use_stdout()


__pack__ = GrpoLarge
