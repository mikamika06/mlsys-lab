def patch_chat_template(template: str) -> str:
    """Patch template to handle non-first system messages."""
    raise NotImplementedError


def render_chat(template: str, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Render chat template with messages."""
    raise NotImplementedError
