import numpy as np

def generate_log_lines():
    lines = [
        "onednn_verbose,info,cpu,convolution,jit:amx,forward,data:f32,g0_mb1_ic64_oc64_ih56_oh56_kh3_sh1_dh1_ph1_pw1,1.23",
        "onednn_verbose,info,cpu,convolution,jit:ref,fallback,data:f32,format_mismatch,g0_mb1_ic3_oc64_ih224_oh224_kh7_sh2_dh1_ph3_pw3,4.56",
        "onednn_verbose,info,cpu,pooling,jit:avx512,forward,data:f32,g0_mb1_ic64_ih56_oh56_kh3_sh1_ph1,0.50"
    ]
    return lines

def generate_expected_parses():
    return [
        {
            "primitive": "convolution",
            "jit": "jit:amx",
            "status": "forward",
            "dims": "g0_mb1_ic64_oc64_ih56_oh56_kh3_sh1_dh1_ph1_pw1",
            "fallback_reason": "none"
        },
        {
            "primitive": "convolution",
            "jit": "jit:ref",
            "status": "fallback",
            "dims": "g0_mb1_ic3_oc64_ih224_oh224_kh7_sh2_dh1_ph3_pw3",
            "fallback_reason": "format_mismatch"
        },
        {
            "primitive": "pooling",
            "jit": "jit:avx512",
            "status": "forward",
            "dims": "g0_mb1_ic64_ih56_oh56_kh3_sh1_ph1",
            "fallback_reason": "none"
        }
    ]

def generate_sweep_records():
    return [
        {"k_val": 16, "kernel": "avx2"},
        {"k_val": 16, "kernel": "avx2"},
        {"k_val": 256, "kernel": "avx512"},
        {"k_val": 256, "kernel": "avx512"},
        {"k_val": 1024, "kernel": "amx"},
        {"k_val": 1024, "kernel": "amx"}
    ]

def generate_resnet_records():
    return [
        {"primitive": "convolution", "time_ms": 600.0},
        {"primitive": "batch_normalization", "time_ms": 150.0},
        {"primitive": "pooling", "time_ms": 50.0},
        {"primitive": "eltwise", "time_ms": 200.0}
    ]
