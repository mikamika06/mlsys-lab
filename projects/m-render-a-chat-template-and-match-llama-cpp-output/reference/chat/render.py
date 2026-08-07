import jinja2


def render_template(template_str, messages, add_generation_prompt=True):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(template_str)
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt)
