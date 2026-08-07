class BenchmarkReporter:
    """Generates benchmark reports for quantization recipes."""

    def __init__(self):
        raise NotImplementedError

    def add_entry(self, recipe_name, size_bytes, ppl, kld, tok_per_sec):
        raise NotImplementedError

    def generate_table(self):
        raise NotImplementedError
