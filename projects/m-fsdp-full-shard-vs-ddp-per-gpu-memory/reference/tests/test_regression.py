import sys

sys.path.insert(0, ".")
from distmem.memory import compute_memory_comparison
from distmem.traces import classify_trace_pattern
from distmem.optimizer import count_graph_breaks


def test_fsdp_memory_less_than_ddp():
    ddp, fsdp = compute_memory_comparison(1000000000, 2, 12, 8)
    assert fsdp < ddp, f"FSDP memory {fsdp} not less than DDP memory {ddp}"


def test_trace_classification_valid():
    res = classify_trace_pattern(["all_gather", "compute", "reduce_scatter"])
    assert res == "fsdp_fully_sharded_pipelined"


def test_graph_breaks_non_negative():
    breaks = count_graph_breaks(True, True)
    assert (
        breaks > 0
    ), "Graph breaks must be greater than zero when hooks and splitting are enabled"
