class ModelConverter:
    def verify_input_contract(self):
        raise NotImplementedError

    def fix_problematic_ops(self):
        raise NotImplementedError

    def predict(self, inputs):
        raise NotImplementedError
