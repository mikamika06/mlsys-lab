import random

CONFIG_TEMPLATES = [
    {
        "peft": {
            "r": 8,
            "target_modules": ["q_proj", "v_proj"],
            "torch_dtype": "float16"
        },
        "shapes": {
            "num_layers": 32,
            "q_proj": (4096, 4096),
            "v_proj": (4096, 4096),
            "k_proj": (4096, 4096),
            "o_proj": (4096, 4096),
        }
    },
    {
        "peft": {
            "r": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "torch_dtype": "float32"
        },
        "shapes": {
            "num_layers": 24,
            "q_proj": (2048, 2048),
            "k_proj": (2048, 2048),
            "v_proj": (2048, 2048),
            "o_proj": (2048, 2048),
        }
    },
    {
        "peft": {
            "r": 64,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
            "torch_dtype": "bfloat16"
        },
        "shapes": {
            "num_layers": 16,
            "gate_proj": (1024, 2816),
            "up_proj": (1024, 2816),
            "down_proj": (2816, 1024),
        }
    },
    {
        "peft": {
            "r": 4,
            "target_modules": ["q_proj", "v_proj"],
            "torch_dtype": "int8"
        },
        "shapes": {
            "num_layers": 48,
            "q_proj": (8192, 8192),
            "v_proj": (8192, 8192),
        }
    },
    {
        "peft": {
            "r": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "torch_dtype": "float16"
        },
        "shapes": {
            "num_layers": 12,
            "q_proj": (1536, 1536),
            "k_proj": (1536, 1536),
            "v_proj": (1536, 1536),
        }
    }
]


def generate_workload(seed=42):
    rng = random.Random(seed)
    adapters = [f"adapter_{i}" for i in range(5)]
    requests = []
    for req_id in range(40):
        req = {
            "id": req_id,
            "adapter_id": rng.choice(adapters) if rng.random() > 0.1 else None,
            "prompt_tokens": rng.randint(10, 100)
        }
        requests.append(req)
    return requests
