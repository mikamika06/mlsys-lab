def parse_gguf_chat_template(metadata):
    return metadata.get("tokenizer.chat_template", "")
