from finetune.memory import compute_training_memory

def compute_memory_ratios(param_counts):
    full_lora_ratios = []
    lora_qlora_ratios = []
    for pc in param_counts:
        m_full = compute_training_memory(pc, "full")
        m_lora = compute_training_memory(pc, "lora")
        m_qlora = compute_training_memory(pc, "qlora")
        full_lora_ratios.append(m_full / m_lora)
        lora_qlora_ratios.append(m_lora / m_qlora)
    return {"full_lora": full_lora_ratios, "lora_qlora": lora_qlora_ratios}
