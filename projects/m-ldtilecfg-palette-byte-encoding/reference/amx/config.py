def encode_tilecfg(tile_specs, palette_id=1, start_row=0):
    buf = bytearray(64)
    buf[0] = palette_id & 0xFF
    buf[1] = start_row & 0xFF

    for t_id, spec in tile_specs.items():
        if not (0 <= t_id <= 7):
            continue
        colsb = spec.get("bytes_per_row", 0)
        rows = spec.get("rows", 0)

        b_idx = 16 + t_id * 2
        buf[b_idx] = colsb & 0xFF
        buf[b_idx + 1] = (colsb >> 8) & 0xFF

        r_idx = 48 + t_id
        buf[r_idx] = rows & 0xFF

    return bytes(buf)


def decode_tilecfg(buffer):
    if len(buffer) != 64:
        raise ValueError("Buffer must be exactly 64 bytes")
    palette_id = buffer[0]
    start_row = buffer[1]

    tiles = {}
    for t_id in range(8):
        b_idx = 16 + t_id * 2
        colsb = buffer[b_idx] | (buffer[b_idx + 1] << 8)
        r_idx = 48 + t_id
        rows = buffer[r_idx]
        if colsb > 0 or rows > 0:
            tiles[t_id] = {"bytes_per_row": colsb, "rows": rows}

    return {"palette_id": palette_id, "start_row": start_row, "tiles": tiles}
