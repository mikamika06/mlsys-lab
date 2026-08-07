import numpy as np


class Quantizer:
    def __init__(self, quant_bits=8):
        self.quant_bits = quant_bits

    def calibrate(self, graph, calibration_dataset):
        calib_stats = {}
        for sample in calibration_dataset:
            curr = sample
            for node in graph.nodes:
                if node.name not in calib_stats:
                    calib_stats[node.name] = {"min": float(np.min(curr)), "max": float(np.max(curr))}
                else:
                    calib_stats[node.name]["min"] = min(calib_stats[node.name]["min"], float(np.min(curr)))
                    calib_stats[node.name]["max"] = max(calib_stats[node.name]["max"], float(np.max(curr)))
                if node.op_type == "relu":
                    curr = np.maximum(curr, 0.0)
        return calib_stats

    def quantize(self, graph, calibration_dataset):
        stats = self.calibrate(graph, calibration_dataset)
        q_graph = graph

        for node in q_graph.nodes:
            if node.weight is not None:
                w = node.weight
                w_min = float(np.min(w))
                w_max = float(np.max(w))
                scale_w = max((w_max - w_min) / 255.0, 1e-7)
                zp_w = int(np.round(-w_min / scale_w)) - 128
                zp_w = int(np.clip(zp_w, -128, 127))

                q_weight = np.clip(np.round(w / scale_w) + zp_w, -128, 127).astype(np.int8)

                node.weight = q_weight
                node.quant_params = {
                    "scale_w": scale_w,
                    "zp_w": zp_w,
                    "act_min": stats.get(node.name, {}).get("min", -1.0),
                    "act_max": stats.get(node.name, {}).get("max", 1.0),
                }

        return q_graph

    def compute_accuracy_loss(self, fp32_output, int8_output):
        return float(np.mean((fp32_output - int8_output) ** 2))
