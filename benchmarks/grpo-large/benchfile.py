from milabench.pack import Package
from milabench.commands import (
    AccelerateAllNodes,
    AccelerateLaunchCommand,
    DockerRunCommand,
    PackCommand,
)
from milabench.system import DockerConfig


# Milabench's `AccelerateLaunchCommand` hard-codes `--zero_stage=2` when the
# bench sets `use_deepspeed: true`. For 72B+ Qwen MoE checkpoints we need
# Zero-3 + CPU offload. We inject the extra accelerate flags after the
# ones milabench emits so they win in argparse's "last value wins" rule.
class AccelerateZero3AllNodes(AccelerateAllNodes):
    def __init__(self, executor, *extra_argv, **kwargs) -> None:
        super().__init__(executor, **kwargs)
        self.extra_argv = extra_argv

    def _wrap(self, executor, rank):
        return AccelerateLaunchCommand(
            executor, rank, *self.extra_argv, **self.options
        )

    def single_node(self):
        ngpu = len(self.executor.pack.config.get("devices", []))
        if ngpu > 1:
            return self._wrap(self.executor, 0)
        return self.executor

    def make_new_node_executor(self, rank, node, base):
        config = base.pack.config
        pack = self.make_new_node_pack(rank, node, base)
        executor = base.copy(pack)
        return DockerRunCommand(
            self._wrap(executor, rank),
            DockerConfig(**config["system"].get("docker", {})),
        )


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
        return AccelerateZero3AllNodes(
            plan,
            "--zero_stage=3",
            "--offload_optimizer_device=cpu",
            "--offload_param_device=cpu",
            "--zero3_init_flag=true",
        ).use_stdout()


__pack__ = GrpoLarge
