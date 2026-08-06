def format_template(style: str, custom_bos: str = "", custom_eos: str = "") -> str:
    """Format a Jinja2 chat template for the given style."""
    raise NotImplementedError


def validate_template(template_str: str) -> bool:
    """Validate Jinja chat template structure."""
    raise NotImplementedError
