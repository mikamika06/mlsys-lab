from coldcache.protocol import ColdCacheProtocol


def verify_cold_execution(protocol: ColdCacheProtocol, requests: list[list[int]]) -> dict:
    """Runs requests and returns verification proof metrics."""
    hits = 0
    generations = []

    for req in requests:
        protocol.invalidate_host_cache()
        gen = protocol.reset_gpu_allocator()
        result = protocol.execute_request(req)
        if result["hit"]:
            hits += 1
        generations.append(gen)

    is_strictly_cold = (hits == 0) and (len(set(generations)) == len(requests))
    return {
        "hits": hits,
        "generations": generations,
        "strictly_cold": is_strictly_cold
    }
