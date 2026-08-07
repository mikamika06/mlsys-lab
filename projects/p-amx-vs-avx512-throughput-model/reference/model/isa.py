class ISAParameters:
    def __init__(self, name: str, vector_width_bits: int, num_accumulators: int,
                 ops_per_instruction: int, tile_config_cost_cycles: int,
                 tile_release_cost_cycles: int, L1_bw_bytes_per_cycle: float):
        self.name = name
        self.vector_width_bits = vector_width_bits
        self.num_accumulators = num_accumulators
        self.ops_per_instruction = ops_per_instruction
        self.tile_config_cost_cycles = tile_config_cost_cycles
        self.tile_release_cost_cycles = tile_release_cost_cycles
        self.L1_bw_bytes_per_cycle = L1_bw_bytes_per_cycle

    def bytes_per_element(self, dtype: str) -> int:
        dmap = {"int8": 1, "bf16": 2, "fp32": 4}
        if dtype not in dmap:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return dmap[dtype]


def get_avx512_params() -> ISAParameters:
    return ISAParameters(
        name="avx512",
        vector_width_bits=512,
        num_accumulators=32,
        ops_per_instruction=64,
        tile_config_cost_cycles=0,
        tile_release_cost_cycles=0,
        L1_bw_bytes_per_cycle=64.0
    )


def get_amx_params() -> ISAParameters:
    return ISAParameters(
        name="amx",
        vector_width_bits=1024,
        num_accumulators=8,
        ops_per_instruction=1024,
        tile_config_cost_cycles=120,
        tile_release_cost_cycles=40,
        L1_bw_bytes_per_cycle=128.0
    )
