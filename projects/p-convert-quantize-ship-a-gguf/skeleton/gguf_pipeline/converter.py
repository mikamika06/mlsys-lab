class GGUFConverter:
    """Converts HuggingFace tensor formats and vocabularies into GGUF model files."""

    def __init__(self, model_config, vocab):
        raise NotImplementedError

    def convert_tensors(self, hf_tensors):
        raise NotImplementedError

    def export_gguf(self, hf_tensors, output_path):
        raise NotImplementedError
