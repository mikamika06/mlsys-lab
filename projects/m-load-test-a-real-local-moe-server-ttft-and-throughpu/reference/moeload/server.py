class MoEServer:
    """Simulates a local MoE server processing prefill and token generation."""

    def __init__(self, num_experts=8, active_experts=2, base_prefill_ms=10.0, gen_ms_per_tok=2.0):
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.base_prefill_ms = base_prefill_ms
        self.gen_ms_per_tok = gen_ms_per_tok

    def process_request(self, prompt_tokens, decode_tokens, concurrency=1):
        expert_factor = 1.0 + (self.active_experts / float(self.num_experts))
        contention_factor = 1.0 + 0.15 * (concurrency - 1)

        prefill_time = self.base_prefill_ms * (prompt_tokens / 100.0) * expert_factor * contention_factor
        gen_time_per_token = self.gen_ms_per_tok * contention_factor

        ttft = prefill_time + gen_time_per_token
        inter_token_latencies = [gen_time_per_token] * max(0, decode_tokens - 1)
        total_time = ttft + sum(inter_token_latencies)

        return {
            "prompt_tokens": prompt_tokens,
            "decode_tokens": decode_tokens,
            "ttft_ms": ttft,
            "inter_token_latencies_ms": inter_token_latencies,
            "total_time_ms": total_time,
            "total_tokens": prompt_tokens + decode_tokens,
        }
