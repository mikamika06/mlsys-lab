from vllmsched.metrics import calculate_throughput, measure_concurrency_sweep
from vllmsched.policy import compare_policies_latency
from vllmsched.scheduler import Request, Scheduler


def gen_test_requests(count):
    reqs = []
    for i in range(count):
        reqs.append(
            Request(
                req_id=i + 1,
                prompt_len=16,
                max_gen_len=20,
                arrival_time=i % 3,
                priority=1,
            )
        )
    return reqs
