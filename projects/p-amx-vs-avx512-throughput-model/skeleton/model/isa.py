class ISAParameters:
    def __init__(self, name: str, vector_width_bits: int, num_accumulators: int,
                 ops_per_instruction: int, tile_config_cost_cycles: int,
                 tile_release_cost_cycles: int, L1_bw_bytes_per_cycle: float):
        raise NotImplementedError

    def bytes_per_element(self, dtype: str) -> int:
        raise NotImplementedError


def get_avx512_params() -> ISAParameters:
    raise NotImplementedError


def get_amx_params() -> ISAParameters:
    raise NotImplementedError
