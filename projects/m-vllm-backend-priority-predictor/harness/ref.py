import random


def generate_logs():
    random.seed(42)
    logs = []
    backends = ["FLASH_ATTN", "FLASHINFER", "XFORMERS", "SDPA"]
    reasons = ["unsupported head dim", "sm version too low", "disabled by user", "missing dependency"]
    for i in range(5):
        chosen = random.choice(backends)
        rejected = random.sample(reasons, 2)
        log_lines = [
            "INFO 00-00 00:00:00 [core.py:100] Found 1 CUDA devices",
            f"INFO 00-00 00:00:00 [attention.py:200] Evaluated backends: {backends}",
            f"INFO 00-00 00:00:00 [attention.py:210] Backend {backends[0]} rejected because {rejected[0]}",
            f"INFO 00-00 00:00:00 [attention.py:210] Backend {backends[1]} rejected because {rejected[1]}",
            f"INFO 00-00 00:00:00 [attention.py:250] Selected attention backend: {chosen}"
        ]
        logs.append("\n".join(log_lines))
    return logs


LOGS = generate_logs()


def parse_log(log_text):
    lines = log_text.split("\n")
    evaluated = []
    rejections = {}
    selected = None
    for line in lines:
        if "Evaluated backends:" in line:
            part = line.split("Evaluated backends:")[1].strip()
            evaluated = [b.strip() for b in part.strip("[]").replace("'", "").split(",")]
        elif "rejected because" in line:
            parts = line.split("Backend ")[1].split(" rejected because ")
            b_name = parts[0].strip()
            reason = parts[1].strip()
            rejections[b_name] = reason
        elif "Selected attention backend:" in line:
            selected = line.split("Selected attention backend:")[1].strip()
    return {"evaluated": evaluated, "rejections": rejections, "selected": selected}


def predict_backend(evaluated, rejections):
    for b in evaluated:
        if b not in rejections:
            return b
    return evaluated[-1]


def get_rejection_reason(backend, rejections):
    return rejections.get(backend, None)
