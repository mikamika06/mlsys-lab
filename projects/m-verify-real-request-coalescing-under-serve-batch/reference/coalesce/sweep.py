import asyncio
from typing import Any, Callable, Dict, List, Tuple

def run_parameter_sweep(
    handler_factory: Callable[[int, float], Callable],
    request_sequence: List[Tuple[float, Any]],
    max_batch_sizes: List[int],
    timeouts: List[float]
) -> List[Dict[str, Any]]:
    records = []
    for mbs in max_batch_sizes:
        for timeout in timeouts:
            record = _evaluate_single(handler_factory, request_sequence, mbs, timeout)
            records.append(record)
    return records

def _evaluate_single(
    handler_factory: Callable[[int, float], Callable],
    request_sequence: List[Tuple[float, Any]],
    max_batch_size: int,
    batch_wait_timeout_s: float
) -> Dict[str, Any]:
    async def _run():
        wrapped_fn = handler_factory(max_batch_size, batch_wait_timeout_s)
        start_t = asyncio.get_running_loop().time()

        async def worker(arrival_delay: float, val: Any):
            await asyncio.sleep(arrival_delay)
            req_start = asyncio.get_running_loop().time()
            res = await wrapped_fn(val)
            req_end = asyncio.get_running_loop().time()
            return (req_end - req_start), res

        tasks = [asyncio.create_task(worker(arr, v)) for arr, v in request_sequence]
        results = await asyncio.gather(*tasks)
        end_t = asyncio.get_running_loop().time()

        latencies = [lat for lat, _ in results]
        total_duration = end_t - start_t
        batch_queue = getattr(wrapped_fn, "__batch_queue__", None)
        num_batches = batch_queue.batch_count if batch_queue else -1

        return {
            "max_batch_size": max_batch_size,
            "batch_wait_timeout_s": batch_wait_timeout_s,
            "total_requests": len(request_sequence),
            "num_batches": num_batches,
            "total_duration_s": total_duration,
            "mean_latency_s": sum(latencies) / len(latencies) if latencies else 0.0,
            "throughput_req_per_s": len(request_sequence) / total_duration if total_duration > 0 else 0.0
        }

    return asyncio.run(_run())
