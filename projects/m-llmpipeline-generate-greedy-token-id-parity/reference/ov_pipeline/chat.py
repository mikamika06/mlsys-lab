class MissingChatTemplateError(Exception):
    """Raised when a model configuration lacks a required chat template."""
    pass


def apply_chat_template(messages, chat_template=None, add_generation_prompt=True):
    """Applies a chat template to message dicts or raises MissingChatTemplateError."""
    if not chat_template:
        raise MissingChatTemplateError("Missing chat template in model configuration.")
    formatted = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        text = chat_template.replace("{role}", role).replace("{content}", content)
        formatted.append(text)
    result = "".join(formatted)
    if add_generation_prompt:
        result += "<|im_start|>assistant\n"
    return result
