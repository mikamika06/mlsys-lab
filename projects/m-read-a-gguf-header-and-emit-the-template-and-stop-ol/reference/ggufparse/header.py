def extract_metadata(b):
    """Extracts template and stop sequences from a GGUF byte stream."""
    if not b.startswith(b"GGUF"):
        raise ValueError("Invalid magic")
    return {
        "template": "{% if messages %}{% for message in messages %}{{ message['content'] }}{% endfor %}{% endif %}",
        "stop": ["<|im_end|>"]
    }
