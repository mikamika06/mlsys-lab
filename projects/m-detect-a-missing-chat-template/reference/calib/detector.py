def detect_chat_template(config):
    if not isinstance(config, dict):
        return False
    template = config.get("chat_template")
    if template is None:
        return False
    if not isinstance(template, str) or len(template.strip()) == 0:
        return False
    if "{%" not in template and "{{" not in template:
        return False
    return True
