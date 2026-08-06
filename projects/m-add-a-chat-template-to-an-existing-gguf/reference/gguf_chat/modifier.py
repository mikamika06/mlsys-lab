def extract_template(reader) -> str | None:
    if "tokenizer.chat_template" not in reader.fields:
        return None
    field = reader.fields["tokenizer.chat_template"]
    return bytes(field.parts[-1]).decode("utf-8")

def set_chat_template(reader, writer, template_str: str):
    backup_field = reader.fields.get("tokenizer.chat_template.backup")
    if backup_field is not None:
        backup_str = bytes(backup_field.parts[-1]).decode("utf-8")
        writer.add_string("tokenizer.chat_template.backup", backup_str)
    else:
        current_field = reader.fields.get("tokenizer.chat_template")
        if current_field is not None:
            current_str = bytes(current_field.parts[-1]).decode("utf-8")
            writer.add_string("tokenizer.chat_template.backup", current_str)
            
    writer.add_string("tokenizer.chat_template", template_str)
