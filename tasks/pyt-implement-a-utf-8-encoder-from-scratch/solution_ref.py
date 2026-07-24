def utf8_encode(text: str) -> bytes:
    out = bytearray()
    for ch in text:
        code = ord(ch)
        if code <= 0x7F:
            out.append(code)
        elif code <= 0x7FF:
            out.append(0xC0 | (code >> 6))
            out.append(0x80 | (code & 0x3F))
        elif code <= 0xFFFF:
            out.append(0xE0 | (code >> 12))
            out.append(0x80 | ((code >> 6) & 0x3F))
            out.append(0x80 | (code & 0x3F))
        else:
            out.append(0xF0 | (code >> 18))
            out.append(0x80 | ((code >> 12) & 0x3F))
            out.append(0x80 | ((code >> 6) & 0x3F))
            out.append(0x80 | (code & 0x3F))
    return bytes(out)
