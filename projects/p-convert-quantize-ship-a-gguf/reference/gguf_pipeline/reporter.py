class BenchmarkReporter:
    """Generates benchmark reports for quantization recipes."""

    def __init__(self):
        self.entries = []

    def add_entry(self, recipe_name, size_bytes, ppl, kld, tok_per_sec):
        self.entries.append({
            "recipe": recipe_name,
            "size_mb": size_bytes / (1024 * 1024),
            "ppl": ppl,
            "kld": kld,
            "tok_per_sec": tok_per_sec
        })

    def generate_table(self):
        headers = ["Recipe", "Size (MB)", "PPL", "KLD", "Tokens/sec"]
        lines = [" | ".join(headers), "|---" * len(headers) + "|"]
        for e in self.entries:
            lines.append(f"{e['recipe']} | {e['size_mb']:.2f} | {e['ppl']:.4f} | {e['kld']:.4f} | {e['tok_per_sec']:.2f}")
        return "\n".join(lines)
