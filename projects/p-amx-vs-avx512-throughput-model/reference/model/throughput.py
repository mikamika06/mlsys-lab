import math
from model.isa import ISAParameters


class ThroughputModel:
    def __init__(self, isa: ISAParameters):
        self.isa = isa

    def compute_tile_dimensions(self, M: int, N: int, K: int, dtype: str) -> dict:
        elem_size = self.isa.bytes_per_element(dtype)
        if self.isa.name == "amx":
            tile_m = 16
            tile_n = 16 if dtype == "fp32" else (32 if dtype == "bf16" else 64)
            tile_k = 16 if dtype == "fp32" else (32 if dtype == "bf16" else 64)
        else:
            vec_elems = self.isa.vector_width_bits // (elem_size * 8)
            tile_m = 4
            tile_n = vec_elems
            tile_k = 4 if dtype == "int8" else (2 if dtype == "bf16" else 1)

        num_tiles_m = math.ceil(M / tile_m)
        num_tiles_n = math.ceil(N / tile_n)
        num_tiles_k = math.ceil(K / tile_k)

        return {
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "num_tiles_m": num_tiles_m,
            "num_tiles_n": num_tiles_n,
            "num_tiles_k": num_tiles_k,
        }

    def predict_cycles(self, M: int, N: int, K: int, dtype: str) -> float:
        elem_size = self.isa.bytes_per_element(dtype)
        t_info = self.compute_tile_dimensions(M, N, K, dtype)

        total_ops = 2.0 * M * N * K
        ops_per_instr = float(self.isa.ops_per_instruction)
        if dtype == "int8":
            ops_per_instr *= 2.0
        elif dtype == "bf16":
            ops_per_instr *= 1.0
        elif dtype == "fp32":
            ops_per_instr *= 0.5

        total_instructions = math.ceil(total_ops / max(1.0, ops_per_instr))
        compute_cycles = total_instructions / self.isa.num_accumulators

        bytes_a = M * K * elem_size
        bytes_b = K * N * elem_size
        bytes_c = M * N * 4
        total_memory_bytes = bytes_a + bytes_b + bytes_c

        memory_cycles = total_memory_bytes / self.isa.L1_bw_bytes_per_cycle

        setup_overhead = float(
            self.isa.tile_config_cost_cycles + self.isa.tile_release_cost_cycles
        )

        return max(compute_cycles, memory_cycles) + setup_overhead

    def predict_gflops(self, M: int, N: int, K: int, dtype: str, clock_ghz: float = 2.0) -> float:
        cycles = self.predict_cycles(M, N, K, dtype)
        if cycles <= 0:
            return 0.0
        total_giga_ops = (2.0 * M * N * K) / 1e9
        seconds = cycles / (clock_ghz * 1e9)
        return total_giga_ops / seconds
