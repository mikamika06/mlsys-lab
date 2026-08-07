import json

def apply_chat_template(messages: list) -> str:
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
    return formatted

def handle_request(req: dict) -> dict:
    messages = req.get("messages", [])
    prompt = apply_chat_template(messages)
    return {"choices": [{"message": {"role": "assistant", "content": f"Echo: {prompt[-20:]}"}}]}
