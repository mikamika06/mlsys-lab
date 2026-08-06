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
