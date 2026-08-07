import re

def parse_log(log_text):
    model = None
    tp = None
    quant = None
    version = None
    m_ver = re.search(r"vLLM engine \(v([0-9\.]+)\)", log_text)
    if m_ver:
        version = m_ver.group(1)
    m_cfg = re.search(r"model='([^']+)', tensor_parallel_size=(\d+), .*?quantization=(\w+)", log_text)
    if m_cfg:
        model = m_cfg.group(1)
        tp = int(m_cfg.group(2))
        quant = m_cfg.group(3)
    return {"model": model, "tensor_parallel_size": tp, "quantization": quant, "version": version}
