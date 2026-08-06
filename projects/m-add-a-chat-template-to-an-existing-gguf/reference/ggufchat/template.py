STYLES = {
    "chatml": "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}",
    "llama3": "{% for message in messages %}{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{'<|start_header_id|>assistant<|end_header_id|>\\n\\n'}}{% endif %}",
    "mistral": "{% for message in messages %}{% if message['role'] == 'user' %}{{'[INST] ' + message['content'] + ' [/INST]'}}{% elif message['role'] == 'assistant' %}{{message['content']}}{% endif %}{% endfor %}",
    "zephyr": "{% for message in messages %}{{'<|' + message['role'] + '|>\\n' + message['content'] + '</s>\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|assistant|>\\n'}}{% endif %}",
}


def format_template(style: str, custom_bos: str = "", custom_eos: str = "") -> str:
    """Format a Jinja2 chat template for the given style."""
    style_key = style.lower().strip()
    if style_key not in STYLES:
        raise ValueError(f"Unsupported template style: {style}")
    base = STYLES[style_key]
    prefix = custom_bos if custom_bos else ""
    suffix = custom_eos if custom_eos else ""
    return prefix + base + suffix


def validate_template(template_str: str) -> bool:
    """Validate Jinja chat template structure."""
    if not isinstance(template_str, str) or not template_str.strip():
        return False
    if "messages" not in template_str:
        return False
    if "message['content']" not in template_str and "message.content" not in template_str:
        return False

    for_starts = template_str.count("{% for ")
    for_ends = template_str.count("{% endfor %}")
    if for_starts != for_ends or for_starts == 0:
        return False

    if_starts = template_str.count("{% if ")
    if_ends = template_str.count("{% endif %}")
    if if_starts != if_ends:
        return False

    return True
