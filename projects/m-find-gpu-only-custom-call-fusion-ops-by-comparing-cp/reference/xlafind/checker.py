def verify_fusion_shapes(hlo_text):
    errors = []
    lines = hlo_text.splitlines()
    for line in lines:
        if "fusion" in line or "custom-call" in line:
            if "shape=" in line and "s32" in line and "f32" in line:
                if "invalid_shape" in line:
                    errors.append(line.strip())
    return errors
