import numpy as np
from benchaudit.detector import audit_benchmark_configs
from benchaudit.reconstruct import reconstruct_cli
from benchaudit.stats import required_sample_size

np.random.seed(42)

AUDIT_CONFIGS = [
    {"config_id": "cfg_0", "num_warmup_requests": 10, "ignore_eos": False, "min_prompt_len": 64, "max_prompt_len": 512, "min_output_len": 16, "max_output_len": 256},
    {"config_id": "cfg_1", "num_warmup_requests": 0, "ignore_eos": False, "min_prompt_len": 64, "max_prompt_len": 512, "min_output_len": 16, "max_output_len": 256},
    {"config_id": "cfg_2", "num_warmup_requests": 5, "ignore_eos": True, "min_prompt_len": 128, "max_prompt_len": 128, "min_output_len": 32, "max_output_len": 128},
    {"config_id": "cfg_3", "num_warmup_requests": 0, "ignore_eos": True, "min_prompt_len": 64, "max_prompt_len": 256, "min_output_len": 16, "max_output_len": 16},
    {"config_id": "cfg_4", "num_warmup_requests": 20, "ignore_eos": False, "min_prompt_len": 100, "max_prompt_len": 100, "min_output_len": 50, "max_output_len": 100},
    {"config_id": "cfg_5", "num_warmup_requests": 10, "ignore_eos": True, "min_prompt_len": 32, "max_prompt_len": 64, "min_output_len": 16, "max_output_len": 32},
    {"config_id": "cfg_6", "num_warmup_requests": 0, "ignore_eos": False, "min_prompt_len": 128, "max_prompt_len": 256, "min_output_len": 64, "max_output_len": 128},
    {"config_id": "cfg_7", "num_warmup_requests": 15, "ignore_eos": False, "min_prompt_len": 64, "max_prompt_len": 128, "min_output_len": 32, "max_output_len": 64},
]

RESULT_METADATA = [
    {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "vllm",
        "endpoint": "/v1/completions",
        "request_rate": 16.0,
        "num_prompts": 1000,
        "dataset_name": "sharegpt",
        "max_concurrency": 64,
        "ignore_eos": False
    },
    {
        "model": "mistralai/Mistral-7B-v0.1",
        "backend": "vllm",
        "endpoint": "/v1/chat/completions",
        "request_rate": "inf",
        "num_prompts": 500,
        "dataset_name": "random",
        "ignore_eos": True
    }
]

TTFT_SAMPLES = np.random.lognormal(mean=2.5, sigma=0.5, size=2000).tolist()
