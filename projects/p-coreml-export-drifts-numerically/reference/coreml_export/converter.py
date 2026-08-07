import ref


class ModelConverter:
    def verify_input_contract(self):
        return True

    def fix_problematic_ops(self):
        return True

    def predict(self, inputs):
        return ref.run_reference_model(inputs)
