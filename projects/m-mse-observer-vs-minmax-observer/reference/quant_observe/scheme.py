def parse_scheme(name: str) -> dict:
    parts = name.split("-")
    dtype = parts[0]
    sym = parts[1] == "sym"
    if dtype.startswith("int"):
        bits = int(dtype[3:])
    else:
        bits = int(dtype[4:])
    return {"bits": bits, "symmetric": sym}
