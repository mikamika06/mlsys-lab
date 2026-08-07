TOOL_PROFILES = [
    {
        "name": "Ollama_v0.1",
        "supported_architectures": ["llama", "mistral", "gemma"],
        "required_keys": ["{arch}.context_length", "{arch}.block_count"],
        "max_context_length": 32768,
        "max_memory_bytes": 16 * 1024 * 1024 * 1024
    },
    {
        "name": "EdgeInference_v2",
        "supported_architectures": ["llama", "phi"],
        "required_keys": ["{arch}.context_length", "general.architecture"],
        "max_context_length": 4096,
        "max_memory_bytes": 4 * 1024 * 1024 * 1024
    }
]

TEST_METADATA = [
    {
        "general.architecture": "llama",
        "llama.context_length": 8192,
        "llama.block_count": 32,
        "general.estimated_memory_bytes": 8 * 1024 * 1024 * 1024
    },
    {
        "general.architecture": "mamba",
        "mamba.context_length": 2048,
        "general.estimated_memory_bytes": 2 * 1024 * 1024 * 1024
    },
    {
        "general.architecture": "phi",
        "phi.context_length": 16384,
        "general.estimated_memory_bytes": 6 * 1024 * 1024 * 1024
    }
]

MODELFILE_PARAMS = {
    "gguf_path": "/models/llama-3-8b-Q4_K_M.gguf",
    "params": {
        "template": "{{ .System }}\nUSER: {{ .Prompt }}\nASSISTANT:",
        "system": "You are a helpful AI assistant.",
        "parameters": {
            "num_ctx": 4096,
            "stop": ["USER:", "ASSISTANT:"],
            "temperature": 0.7
        }
    }
}

TENSOR_LIST = [
    {"name": "token_embd.weight", "shape": [32000, 4096], "is_embedding": True},
    {"name": "blk.0.attn_q.weight", "shape": [4096, 4096]},
    {"name": "blk.0.attn_k.weight", "shape": [1024, 4096]},
    {"name": "blk.0.attn_v.weight", "shape": [1024, 4096]},
    {"name": "output_norm.weight", "shape": [4096], "is_norm": True}
]


def oracle_check_tool_compatibility(metadata: dict, tool_profile: dict) -> dict:
    reasons = []
    arch = metadata.get("general.architecture", "")
    supported_archs = tool_profile.get("supported_architectures", [])
    if supported_archs and arch not in supported_archs:
        reasons.append(f"Unsupported architecture: {arch}")

    req_keys = tool_profile.get("required_keys", [])
    for key in req_keys:
        formatted_key = key.format(arch=arch) if "{arch}" in key else key
        if formatted_key not in metadata:
            reasons.append(f"Missing required key: {formatted_key}")

    max_ctx = tool_profile.get("max_context_length")
    if max_ctx is not None:
        ctx_key = f"{arch}.context_length"
        model_ctx = metadata.get(ctx_key, metadata.get("general.context_length", 0))
        if model_ctx > max_ctx:
            reasons.append(f"Model context length ({model_ctx}) exceeds maximum supported ({max_ctx})")

    max_mem = tool_profile.get("max_memory_bytes")
    if max_mem is not None:
        est_mem = metadata.get("general.estimated_memory_bytes", 0)
        if est_mem > max_mem:
            reasons.append(f"Estimated memory ({est_mem}) exceeds tool limit ({max_mem})")

    return {
        "compatible": len(reasons) == 0,
        "reasons": reasons
    }


def oracle_generate_modelfile(gguf_path: str, params: dict) -> str:
    lines = [f"FROM {gguf_path}"]

    template = params.get("template")
    if template:
        lines.append(f'TEMPLATE """{template}"""')

    system = params.get("system")
    if system:
        lines.append(f'SYSTEM """{system}"""')

    parameters = params.get("parameters", {})
    for k, v in sorted(parameters.items()):
        if isinstance(v, bool):
            v_str = "true" if v else "false"
        elif isinstance(v, list):
            for item in v:
                lines.append(f'PARAMETER {k} "{item}"')
            continue
        else:
            v_str = str(v)
        lines.append(f"PARAMETER {k} {v_str}")

    adapter = params.get("adapter")
    if adapter:
        lines.append(f"ADAPTER {adapter}")

    return "\n".join(lines) + "\n"


def oracle_predict_quant_file_size(tensor_infos: list[dict], quant_type: str, alignment: int = 32) -> int:
    block_sizes = {"F32": 1, "F16": 1, "Q8_0": 32, "Q4_K_M": 256, "Q4_0": 32, "Q2_K": 256}
    bytes_per_block = {"F32": 4, "F16": 2, "Q8_0": 34, "Q4_K_M": 144, "Q4_0": 18, "Q2_K": 84}

    offset = 1024
    for t in tensor_infos:
        numel = 1
        for d in t["shape"]:
            numel *= d

        if t.get("is_embedding") or t.get("is_norm"):
            t_quant = "F16"
        else:
            t_quant = quant_type

        bs = block_sizes[t_quant]
        bpb = bytes_per_block[t_quant]

        n_blocks = (numel + bs - 1) // bs
        raw_bytes = n_blocks * bpb

        pad = (alignment - (offset % alignment)) % alignment
        offset += pad + raw_bytes

    return offset
