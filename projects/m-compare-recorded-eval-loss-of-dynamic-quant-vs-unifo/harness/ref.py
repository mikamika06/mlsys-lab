import re

LOG_DATA = {
  "dynamic": [
    {"step": 100, "eval_loss": 2.450},
    {"step": 200, "eval_loss": 2.120},
    {"step": 300, "eval_loss": 1.950},
    {"step": 400, "eval_loss": 1.820}
  ],
  "uniform": [
    {"step": 100, "eval_loss": 2.500},
    {"step": 200, "eval_loss": 2.200},
    {"step": 300, "eval_loss": 2.050},
    {"step": 400, "eval_loss": 1.900}
  ]
}

MOE_CONFIG = {
  "num_layers": 32,
  "hidden_size": 4096,
  "num_experts": 8,
  "num_experts_per_tok": 2,
  "expert_intermediate_size": 11008,
  "recorded_vram_gb": 48.5
}

INSTALLER_TRANSCRIPT = """
[INFO] Reading package lists...
[INFO] Building dependency tree...
[INSTALL] Downloading torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl
[INSTALL] Successfully installed torch-2.1.2+cu121
[INSTALL] Downloading unsloth-2024.1-py3-none-any.whl
[INSTALL] Successfully installed unsloth-2024.1
[INSTALL] Downloading triton-2.1.0-cp310-cp310-linux_x86_64.whl
[INSTALL] Successfully installed triton-2.1.0
[INSTALL] Downloading bitsandbytes-0.41.1-py3-none-any.whl
[INSTALL] Successfully installed bitsandbytes-0.41.1
"""

def compare_eval_loss(dyn_logs, uni_logs):
  d_map = {item["step"]: item["eval_loss"] for item in dyn_logs}
  u_map = {item["step"]: item["eval_loss"] for item in uni_logs}
  common_steps = sorted(set(d_map.keys()) & set(u_map.keys()))
  if not common_steps:
    return 0.0
  rel_diffs = [abs(d_map[s] - u_map[s]) / u_map[s] for s in common_steps]
  return sum(rel_diffs) / len(rel_diffs)

def reconcile_moe_vram(config):
  layers = config["num_layers"]
  h = config["hidden_size"]
  e = config["num_experts"]
  i = config["expert_intermediate_size"]
  expert_params = layers * e * (3 * h * i)
  non_expert_params = layers * (4 * h * h + 4 * h * h)
  total_params = expert_params + non_expert_params
  bf16_size_gb = (total_params * 2) / (1024 ** 3)
  recorded_vram = config["recorded_vram_gb"]
  overhead_gb = recorded_vram - bf16_size_gb
  return {"bf16_size_gb": bf16_size_gb, "overhead_gb": overhead_gb}

def parse_installer_transcript(transcript):
  pattern = r"Successfully installed\s+([a-zA-Z0-9_\-]+)-([0-9\.\+a-zA-Z]+)"
  matches = re.findall(pattern, transcript)
  return dict(matches)
