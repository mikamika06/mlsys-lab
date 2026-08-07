import re


def parse_startup_log(log_text):
    """Parse vLLM startup log text into structured engine metadata."""
    result = {
        "model": None,
        "tp_size": 1,
        "quantization": None,
        "num_gpu_blocks": None,
        "warnings": [],
        "errors": []
    }

    for line in log_text.splitlines():
        if "WARNING" in line or "WARN" in line:
            msg = line.split("WARNING")[-1].split("WARN")[-1].strip(": ")
            result["warnings"].append(msg)
        if "ERROR" in line:
            msg = line.split("ERROR")[-1].strip(": ")
            result["errors"].append(msg)

        m_model = re.search(r"Initializing an EngineArgs \(model='([^']+)'", line)
        if m_model:
            result["model"] = m_model.group(1)

        m_tp = re.search(r"tensor_parallel_size=(\d+)", line)
        if m_tp:
            result["tp_size"] = int(m_tp.group(1))

        m_quant = re.search(r"quantization='([^']+)'", line)
        if m_quant and m_quant.group(1) != "None":
            result["quantization"] = m_quant.group(1)

        m_blocks = re.search(r"# GPU blocks:\s*(\d+)", line)
        if m_blocks:
            result["num_gpu_blocks"] = int(m_blocks.group(1))

    return result
