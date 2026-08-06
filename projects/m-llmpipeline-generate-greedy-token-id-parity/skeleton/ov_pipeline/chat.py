class MissingChatTemplateError(Exception):
    pass


def apply_chat_template(messages, chat_template=None, add_generation_prompt=True):
    raise NotImplementedError
