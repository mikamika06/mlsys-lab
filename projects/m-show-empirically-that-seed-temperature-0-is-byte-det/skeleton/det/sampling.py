def check_determinism(seed, temperature, prompt_tokens):
    raise NotImplementedError

def break_determinism_state(seed, prompt_tokens):
    raise NotImplementedError

def recover_sampling_params(logits, output_tokens):
    raise NotImplementedError
