import ref


def check(workdir):
    from prefixhash.surgery import optimize_template
    from prefixhash.lora import build_hash_key
    
    surgery_cases = ref.get_surgery_cases()
    surgery_ok = 0
    for tokens, ts in surgery_cases:
        got = optimize_template(tokens, ts)
        want = [t for t in tokens if t != ts] + [ts] if ts in tokens else list(tokens)
        if got == want:
            surgery_ok += 1
            
    lora_cases = ref.get_lora_cases()
    lora_ok = 0
    for block, lora, salt in lora_cases:
        got = build_hash_key(block, lora, salt)
        want = (block, lora, salt)
        if got == want:
            lora_ok += 1
            
    return {
        "surgery_matched": 1.0 if surgery_ok == len(surgery_cases) else 0.0,
        "lora_keys_matched": 1.0 if lora_ok == len(lora_cases) else 0.0
    }
