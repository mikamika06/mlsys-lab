import sys

sys.path.insert(0, ".")
from vllmsched.policy import compare_policies_latency
from vllmsched.scheduler import Request


def test_priority_prevents_inversion():
    reqs = [
        Request(
            req_id=1,
            prompt_len=16,
            max_gen_len=10,
            arrival_time=0,
            priority=1,
        ),
        Request(
            req_id=2,
            prompt_len=16,
            max_gen_len=10,
            arrival_time=0,
            priority=1,
        ),
        Request(
            req_id=3,
            prompt_len=16,
            max_gen_len=10,
            arrival_time=2,
            priority=10,
        ),
    ]

    res = compare_policies_latency(
        requests=reqs,
        num_blocks=4,
        block_size=16,
        max_tokens=32,
        target_req_id=3,
    )

    assert res["priority_latency"] < res["fcfs_latency"], (
        f"Priority latency ({res['priority_latency']}) must be strictly lower "
        f"than FCFS latency ({res['fcfs_latency']}) under priority inversion."
    )
