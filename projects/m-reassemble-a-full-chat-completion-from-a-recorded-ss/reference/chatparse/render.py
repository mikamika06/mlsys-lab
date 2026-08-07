import jinja2

def render_chat_template(template_str, messages, add_generation_prompt=True):
    env = jinja2.Environment()
    template = env.from_string(template_str)
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt)
