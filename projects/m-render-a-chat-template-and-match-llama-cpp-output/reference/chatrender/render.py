import re
import jinja2


def patch_chat_template(template: str) -> str:
    """Patch template to handle non-first system messages."""
    patched = template
    patched = re.sub(
        r"messages\[0\]\['role'\]\s*==\s*'system'",
        "messages[0]['role'] == 'system' or message['role'] == 'system'",
        patched
    )
    patched = re.sub(
        r'messages\[0\]\["role"\]\s*==\s*"system"',
        'messages[0]["role"] == "system" or message["role"] == "system"',
        patched
    )
    return patched


def render_chat(template: str, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Render chat template with messages."""
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    t = env.from_string(template)
    try:
        return t.render(messages=messages, add_generation_prompt=add_generation_prompt)
    except Exception:
        patched = patch_chat_template(template)
        t_patched = env.from_string(patched)
        return t_patched.render(messages=messages, add_generation_prompt=add_generation_prompt)
