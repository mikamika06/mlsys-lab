from .predictor import PeakPredictor

class MemoryAdvisor:
    def __init__(self, predictor=None):
        self.predictor = predictor or PeakPredictor()

    def analyze(self, config: dict) -> dict:
        gpu_capacity = float(config.get("gpu_memory_bytes", 80 * 1024**3))
        pred_bytes = self.predictor.predict_peak_bytes(config)
        fits = pred_bytes <= gpu_capacity
        return {
            "predicted_bytes": pred_bytes,
            "gpu_capacity_bytes": gpu_capacity,
            "fits": fits,
            "headroom_bytes": gpu_capacity - pred_bytes
        }

    def suggest_fixes(self, config: dict) -> list:
        analysis = self.analyze(config)
        if analysis["fits"]:
            return []

        suggestions = []
        cfg = dict(config)

        if cfg.get("activation_checkpointing", "none") == "none":
            cfg_opt = dict(cfg)
            cfg_opt["activation_checkpointing"] = "full"
            if self.analyze(cfg_opt)["fits"]:
                suggestions.append({
                    "action": "enable_activation_checkpointing",
                    "new_config": cfg_opt
                })

        if cfg.get("micro_batch_size", 1) > 1:
            mbs = cfg.get("micro_batch_size", 1)
            gas = cfg.get("grad_accum_steps", 1)
            cfg_opt = dict(cfg)
            cfg_opt["micro_batch_size"] = max(1, mbs // 2)
            cfg_opt["grad_accum_steps"] = gas * (mbs // cfg_opt["micro_batch_size"])
            if self.analyze(cfg_opt)["fits"]:
                suggestions.append({
                    "action": "reduce_micro_batch_size",
                    "new_config": cfg_opt
                })

        if cfg.get("zero_stage", 0) < 3:
            cfg_opt = dict(cfg)
            cfg_opt["zero_stage"] = 3
            if self.analyze(cfg_opt)["fits"]:
                suggestions.append({
                    "action": "increase_zero_stage",
                    "new_config": cfg_opt
                })

        if not cfg.get("offload_optimizer", False):
            cfg_opt = dict(cfg)
            cfg_opt["offload_optimizer"] = True
            if self.analyze(cfg_opt)["fits"]:
                suggestions.append({
                    "action": "enable_optimizer_offload",
                    "new_config": cfg_opt
                })

        if not suggestions:
            cfg_opt = dict(cfg)
            cfg_opt["activation_checkpointing"] = "full"
            cfg_opt["zero_stage"] = 3
            cfg_opt["offload_optimizer"] = True
            mbs = cfg_opt.get("micro_batch_size", 1)
            if mbs > 1:
                cfg_opt["micro_batch_size"] = 1
                cfg_opt["grad_accum_steps"] = cfg_opt.get("grad_accum_steps", 1) * mbs
            suggestions.append({
                "action": "aggressive_optimization",
                "new_config": cfg_opt
            })

        return suggestions
