class ModelCompiler:
    def __init__(self, model):
        raise NotImplementedError

    def localize_control_flow(self):
        raise NotImplementedError

    def translate_branches(self):
        raise NotImplementedError

    def set_dynamic_shapes(self, shapes, constraints):
        raise NotImplementedError

    def verify_equivalence(self, test_inputs):
        raise NotImplementedError

    def export(self):
        raise NotImplementedError
