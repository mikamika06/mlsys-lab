class MoEServer:
    """Simulates a local MoE server processing prefill and token generation."""

    def __init__(self, num_experts=8, active_experts=2, base_prefill_ms=10.0, gen_ms_per_tok=2.0):
        raise NotImplementedError

    def process_request(self, prompt_tokens, decode_tokens, concurrency=1):
        raise NotImplementedError
