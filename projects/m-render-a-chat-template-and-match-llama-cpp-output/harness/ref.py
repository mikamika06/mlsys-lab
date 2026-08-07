import struct


def make_fixture(template: str) -> bytes:
    key = b"tokenizer.chat_template"
    encoded = template.encode("utf-8")
    return key + struct.pack("<I", len(encoded)) + encoded


def render_chat(template: str, messages: list) -> str:
    out = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        out.append(f"<|start|>{role}\n{content}<|end|>\n")
    return "".join(out)


def patch_messages(messages: list) -> list:
    sys_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    return sys_msgs + other_msgs
