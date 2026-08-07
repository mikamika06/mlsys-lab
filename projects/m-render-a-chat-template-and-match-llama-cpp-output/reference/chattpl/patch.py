def patch_messages(messages: list) -> list:
    sys_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    return sys_msgs + other_msgs
