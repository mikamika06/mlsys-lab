import json
import struct

WIDTH = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
         "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
         "U32": 4, "I32": 4, "F32": 4,
         "U64": 8, "I64": 8, "F64": 8}


class SafetensorsError(Exception):
    pass


def parse_header(blob):
    if len(blob) < 8:
        raise SafetensorsError("file shorter than a header length")
    n = struct.unpack_from("<Q", blob, 0)[0]
    if 8 + n > len(blob):
        raise SafetensorsError("header of %d bytes does not fit in %d" % (n, len(blob)))
    try:
        head = json.loads(blob[8:8 + n].decode("utf-8"))
    except Exception as e:
        raise SafetensorsError("header is not JSON: %s" % e) from None
    if not isinstance(head, dict):
        raise SafetensorsError("header is not an object")
    return {"header_bytes": n, "data_start": 8 + n, "header": head,
            "metadata": head.get("__metadata__", {})}


def entries(blob):
    parsed = parse_header(blob)
    out = []
    for name, rec in parsed["header"].items():
        if name == "__metadata__":
            continue
        dtype = rec["dtype"]
        shape = [int(x) for x in rec["shape"]]
        start, end = (int(x) for x in rec["data_offsets"])
        elems = 1
        for d in shape:
            elems *= d
        out.append({
            "name": name, "dtype": dtype, "shape": shape,
            "elements": elems,
            "width": WIDTH.get(dtype, 0),
            "declared_bytes": end - start,
            "expected_bytes": elems * WIDTH.get(dtype, 0),
            "relative_offsets": [start, end],
            "absolute_offsets": [parsed["data_start"] + start,
                                 parsed["data_start"] + end],
        })
    out.sort(key=lambda e: e["relative_offsets"][0])
    return {"data_start": parsed["data_start"], "metadata": parsed["metadata"],
            "tensors": out}


def validate(blob):
    problems = []
    try:
        parsed = entries(blob)
    except SafetensorsError as e:
        return ["header: %s" % e]
    size = len(blob)
    cursor = 0
    for t in parsed["tensors"]:
        if t["width"] == 0:
            problems.append("%s: unknown dtype %s" % (t["name"], t["dtype"]))
        elif t["declared_bytes"] != t["expected_bytes"]:
            problems.append(
                "%s: declares %d bytes, %s%s needs %d"
                % (t["name"], t["declared_bytes"], t["dtype"], t["shape"],
                   t["expected_bytes"]))
        start, end = t["relative_offsets"]
        if start != cursor:
            problems.append("%s: starts at %d, previous tensor ended at %d"
                            % (t["name"], start, cursor))
        if parsed["data_start"] + end > size:
            problems.append("%s: data ends at %d, file is %d bytes"
                            % (t["name"], parsed["data_start"] + end, size))
        cursor = max(cursor, end)
    return problems
