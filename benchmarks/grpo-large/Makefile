# Use global base if possible
ifndef MILABENCH_BASE
	MILABENCH_BASE="base"
endif

export MILABENCH_BASE

BENCH_NAME=grpo-large
MILABENCH_CONFIG=dev.yaml
MILABENCH_ARGS=--config $(MILABENCH_CONFIG) --base $(MILABENCH_BASE)

all:
	install prepare dev

install:
	milabench install $(MILABENCH_ARGS) --force

prepare:
	milabench prepare $(MILABENCH_ARGS)

tests: install prepare
	milabench run $(MILABENCH_ARGS)

dev:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-dev

nodes:
	milabench run $(MILABENCH_ARGS) --select $(BENCH_NAME)-nodes
