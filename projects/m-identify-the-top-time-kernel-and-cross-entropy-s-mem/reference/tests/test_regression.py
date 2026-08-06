import sys

sys.path.insert(0, ".")
from profalyze.kernels import top_time_kernel
from profalyze.memory import cross_entropy_memory_share
import ref


def test_top_kernel_is_valid():
    trace = ref.TRACES[0]
    top = top_time_kernel(trace)
    assert top in [
        "triton_fused_attention_kernel",
        "triton_gemm_kernel",
        "cross_entropy_loss",
        "elementwise_add",
    ]


def test_memory_share_bounds():
    trace = ref.TRACES[0]
    share = cross_entropy_memory_share(trace)
    assert 0.0 <= share <= 1.0
