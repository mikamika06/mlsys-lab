import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from prefix_cache.cache_engine import PrefixCacheEngine
    from prefix_cache.metrics import CacheMetricsTracker

    out = {"hit_rate_matched": 0.0, "ttft_speedup_valid": 0.0}

    ref_engine_off = ref.ReferenceCacheEngine(ref.BLOCK_SIZE, ref.TOTAL_BLOCKS)
    ref_engine_on = ref.ReferenceCacheEngine(ref.BLOCK_SIZE, ref.TOTAL_BLOCKS)
    ref_tracker = ref.ReferenceMetricsTracker()

    learner_engine_on = PrefixCacheEngine(ref.BLOCK_SIZE, ref.TOTAL_BLOCKS)
    learner_tracker = CacheMetricsTracker()

    total_ttft_off = 0.0
    total_ttft_on = 0.0

    for req in ref.DATASET:
        _, ttft_off = ref_engine_off.simulate_request(req)
        ref_engine_off.hash_to_block_id.clear()
        ref_engine_off.free_blocks = list(range(ref.TOTAL_BLOCKS))

        cached_ref, ttft_on_ref = ref_engine_on.simulate_request(req)
        ref_tracker.record_request(len(req), cached_ref, ttft_on_ref)

        cached_learner, ttft_on_learner = learner_engine_on.simulate_request(
            req
        )
        learner_tracker.record_request(len(req), cached_learner, ttft_on_learner)

        total_ttft_off += ttft_off
        total_ttft_on += ttft_on_learner

    ref_hit_rate = ref_tracker.get_hit_rate()
    learner_hit_rate = learner_tracker.get_hit_rate()

    if abs(ref_hit_rate - learner_hit_rate) < 1e-6:
        out["hit_rate_matched"] = 1.0
    else:
        out[
            "_note"
        ] = f"Hit rate mismatch: ref={ref_hit_rate}, learner={learner_hit_rate}"

    if total_ttft_on < total_ttft_off and total_ttft_on > 0.0:
        out["ttft_speedup_valid"] = 1.0

    return out
