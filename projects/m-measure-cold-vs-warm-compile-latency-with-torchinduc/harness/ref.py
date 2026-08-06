"""Reference module helper."""
import tempfile
from compcache.measurement import measure_compile_latencies
from compcache.invalidation import check_cache_behavior
from compcache.mega import run_and_save_cache, verify_zero_recompiles
