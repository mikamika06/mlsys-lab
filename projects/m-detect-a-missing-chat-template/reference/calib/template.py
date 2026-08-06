def check_chat_template(samples, control_tokens, role_markers):
    if not samples:
        return False

    valid_count = 0
    for sample in samples:
        has_control = any(ct in sample for ct in control_tokens)
        has_role = any(rm in sample for rm in role_markers)
        if has_control and has_role:
            valid_count += 1

    return (valid_count / len(samples)) >= 0.8
