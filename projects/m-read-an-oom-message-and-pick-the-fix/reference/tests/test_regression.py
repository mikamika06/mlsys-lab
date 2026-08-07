import sys
sys.path.insert(0, ".")
from triage.parser import parse_oom_message, pick_fix
from triage.allocator import tune_max_split_size
from triage.metrics import track_allocator_loop


def test_parser_extracts_correct_bytes():
    msg = "torch.cuda.OutOfMemoryError: Tried to allocate 512.00 MiB (GPU 0; 15.78 GiB total capacity; 10.00 GiB already allocated; 400.00 MiB free; 14.00 GiB reserved)"
    res = parse_oom_message(msg)
    assert res["requested_bytes"] == 512 * 1024 * 1024
    assert res["allocated_bytes"] == int(10.00 * 1024**3)


def test_pick_fix_returns_valid_action():
    msg = "Tried to allocate 1.00 GiB... 12.00 GiB already allocated; 15.00 GiB reserved in total by PyTorch. max_split_size_mb"
    fix = pick_fix(msg)
    assert fix in ("set_max_split_size", "empty_cache", "reduce_batch_size")


def test_tune_max_split_size_returns_candidate():
    trace = [(1, 1024), (-1, 0), (2, 2048)]
    candidates = [32, 64, 128]
    best = tune_max_split_size(trace, candidates)
    assert best in candidates


def test_track_allocator_loop_metrics():
    res = track_allocator_loop([100, 200], [1])
    assert res["peak_allocated"] == 300
    assert res["final_reserved"] == 300
