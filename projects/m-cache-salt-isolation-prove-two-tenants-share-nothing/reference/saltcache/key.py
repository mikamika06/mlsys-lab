def build_key_tuple(tenant_salt, lora_id, block_tokens):
    return (tenant_salt, lora_id, tuple(block_tokens))
