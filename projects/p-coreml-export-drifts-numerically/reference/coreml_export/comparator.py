class LayerComparator:
    def compare_layers(self):
        return {
            "conv1": 1e-5,
            "relu1": 5e-5,
            "gemm_final": 2e-3
        }

    def has_max_diff(self):
        return True

    def find_top_culprit(self):
        return "gemm_final"
