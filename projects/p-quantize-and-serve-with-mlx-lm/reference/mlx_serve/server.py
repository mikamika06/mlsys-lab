import ref

def format_chat(messages: list) -> str:
    return ref.apply_chat_template(messages)
