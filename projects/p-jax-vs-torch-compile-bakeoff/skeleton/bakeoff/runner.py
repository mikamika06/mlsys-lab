class BakeoffRunner:
    def __init__(self, model_cfg):
        raise NotImplementedError

    def compile_and_run(self, stack_name, inputs):
        raise NotImplementedError

    def evaluate_dynamic(self, stack_name, shape_list):
        raise NotImplementedError

    def export_artifact(self, stack_name):
        raise NotImplementedError

    def compute_intervals(self, runs_a, runs_b):
        raise NotImplementedError

    def recommend(self, workload_type):
        raise NotImplementedError
