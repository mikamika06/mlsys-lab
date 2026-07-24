SIZE = {"bool":1,"char":1,"short":2,"int":4,"long":8,"long long":8,
        "float":4,"double":8,"pointer":8}

def struct_size(fields):
    """Size of a C struct with these fields under LP64 natural alignment + padding."""
    off = 0; maxa = 1
    for t in fields:
        s = 8 if t.endswith("*") else SIZE[t]
        a = s
        off += (-off) % a          # pad to alignment
        off += s
        maxa = max(maxa, a)
    return off + (-off) % maxa      # tail padding to struct alignment
