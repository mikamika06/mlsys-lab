def identify_library(file_list):
    has_quantize_config = "quantize_config.json" in file_list
    has_autogptq_wheel = any("autogptq" in f.lower() for f in file_list)
    has_gptqmodel_meta = any("gptqmodel" in f.lower() for f in file_list)
    has_safetensors = any(f.endswith(".safetensors") for f in file_list)
    has_bin = any(f.endswith(".bin") for f in file_list)

    if has_gptqmodel_meta or (has_quantize_config and not has_autogptq_wheel and has_safetensors):
        return "gptqmodel"
    if has_autogptq_wheel or has_quantize_config or has_bin:
        return "autogptq"
    return "transformers"
