import hashlib
import json
from inductor_audit.config import CompilerConfig


class InductorAuditCompiler:
    def __init__(self, config=None):
        self.config = config if config is not None else CompilerConfig()
        self.cache = {}
        self.cold_start_count = 0
        self.cache_hit_count = 0

    def _hash_graph(self, ops, input_shape):
        key_data = {
            "ops": ops,
            "shape": list(input_shape),
            "config": self.config.to_dict(),
        }
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def compile_graph(self, ops, input_shape):
        key = self._hash_graph(ops, input_shape)
        if key in self.cache:
            self.cache_hit_count += 1
            return self.cache[key]

        self.cold_start_count += 1
        num_elements = 1
        for d in input_shape:
            num_elements *= d

        fused_groups = []
        nodes = []

        if self.config.enable_fusion and num_elements >= self.config.min_fusion_size:
            curr_group = []
            for op in ops:
                if op.get("type") == "reduction" or op.get("stride_mismatch") or op.get("shape_sensitive"):
                    if curr_group:
                        fused_groups.append(curr_group)
                        curr_group = []
                    fused_groups.append([op["name"]])
                else:
                    curr_group.append(op["name"])
            if curr_group:
                fused_groups.append(curr_group)
        else:
            for op in ops:
                fused_groups.append([op["name"]])

        kernel_code_lines = ["# Generated Triton Kernel Code"]
        for group in fused_groups:
            group_fused = len(group) > 1
            kernel_code_lines.append(f"@triton.jit\ndef kernel_{'_'.join(group)}():")
            kernel_code_lines.append(f"    # ops: {', '.join(group)}")
            for name in group:
                op_obj = next((o for o in ops if o["name"] == name), {})
                nodes.append({
                    "name": name,
                    "op_type": op_obj.get("type", "pointwise"),
                    "fused": group_fused,
                    "fused_group": group,
                    "stride_mismatch": op_obj.get("stride_mismatch", False),
                    "shape_sensitive": op_obj.get("shape_sensitive", False),
                })

        code = "\n".join(kernel_code_lines)
        num_kernels = len(fused_groups)

        base_latency = 100.0 if num_elements < 16 else 500.0
        if self.config.enable_fusion:
            latency = base_latency / (2.0 if self.config.max_autotune else 1.5)
        else:
            latency = base_latency * 1.5

        result = {
            "code": code,
            "kernels": num_kernels,
            "graph": {"nodes": nodes},
            "latency_us": latency,
            "cached": False,
        }
        self.cache[key] = result
        return result

    def run_benchmark(self, compiled_kernel, input_shape):
        num_elements = 1
        for d in input_shape:
            num_elements *= d
        eager_latency = 200.0 if num_elements < 16 else 1000.0
        comp_latency = compiled_kernel["latency_us"]
        return {
            "eager_us": eager_latency,
            "compiled_us": comp_latency,
            "speedup": eager_latency / comp_latency,
        }

    def clear_cache(self):
        self.cache.clear()
        self.cold_start_count = 0
        self.cache_hit_count = 0
