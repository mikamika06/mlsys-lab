from trt_builder.compat import compute_compatibility_hash


class PortabilityMatrix:
    def __init__(self):
        self.matrix = {}

    def add_entry(self, source_env, target_env, compatible, reason=""):
        src_hash = compute_compatibility_hash(source_env)
        tgt_hash = compute_compatibility_hash(target_env)
        self.matrix[(src_hash, tgt_hash)] = {
            "compatible": bool(compatible),
            "reason": reason,
            "source_env": source_env,
            "target_env": target_env,
        }

    def can_deploy(self, source_env, target_env):
        src_hash = compute_compatibility_hash(source_env)
        tgt_hash = compute_compatibility_hash(target_env)

        if (src_hash, tgt_hash) in self.matrix:
            return self.matrix[(src_hash, tgt_hash)]["compatible"]

        return src_hash == tgt_hash

    def generate_report(self):
        report = []
        for (src_hash, tgt_hash), entry in self.matrix.items():
            report.append({
                "src_hash": src_hash,
                "tgt_hash": tgt_hash,
                "compatible": entry["compatible"],
                "reason": entry["reason"],
            })
        return report
