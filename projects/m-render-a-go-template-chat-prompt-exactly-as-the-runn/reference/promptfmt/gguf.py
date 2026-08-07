def recover_chat_template(metadata):
    return metadata.get("tokenizer.chat_template", "")

def compare_with_ollama(recovered, show_output):
    return recovered.strip() == show_output.strip()
