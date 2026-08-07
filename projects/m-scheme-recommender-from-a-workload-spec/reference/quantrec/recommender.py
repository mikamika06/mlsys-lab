def recommend_scheme(spec):
    bs = spec["batch_size"]
    hidden = spec["hidden_size"]
    intermediate = spec["intermediate_size"]

    w8a8_bytes = (hidden * intermediate * 2.0) * 0.5 + (hidden * bs * 2.0)
    w4a16_bytes = (hidden * intermediate * 2.0) * 0.25 + (hidden * bs * 4.0)

    if w4a16_bytes < w8a8_bytes and bs <= 8:
        return "W4A16"
    return "W8A8"
