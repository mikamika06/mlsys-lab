import re
from typing import Tuple

VALID_DATA_TYPES = {
    "TYPE_INVALID", "TYPE_BOOL", "TYPE_UINT8", "TYPE_UINT16", "TYPE_UINT32",
    "TYPE_UINT64", "TYPE_INT8", "TYPE_INT16", "TYPE_INT32", "TYPE_INT64",
    "TYPE_FP16", "TYPE_FP32", "TYPE_FP64", "TYPE_STRING", "TYPE_BF16"
}


def validate_pbtxt(pbtxt_str: str) -> Tuple[bool, str]:
    if not isinstance(pbtxt_str, str) or not pbtxt_str.strip():
        return False, "EMPTY_CONFIG"

    lines = [line.strip() for line in pbtxt_str.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return False, "EMPTY_CONFIG"

    has_name = False
    has_platform_or_backend = False
    max_batch_size = 0

    in_input = False
    in_output = False
    current_io = {}
    inputs = []
    outputs = []

    for line in lines:
        if line.startswith("name:"):
            val = line.split(":", 1)[1].strip().strip('"')
            if val:
                has_name = True
        elif line.startswith("platform:") or line.startswith("backend:"):
            val = line.split(":", 1)[1].strip().strip('"')
            if val:
                has_platform_or_backend = True
        elif line.startswith("max_batch_size:"):
            try:
                val = int(line.split(":", 1)[1].strip())
                if val < 0:
                    return False, "INVALID_MAX_BATCH_SIZE"
                max_batch_size = val
            except ValueError:
                return False, "INVALID_MAX_BATCH_SIZE"
        elif line.startswith("input [") or line.startswith("input{") or line == "input":
            in_input = True
            current_io = {}
        elif line.startswith("output [") or line.startswith("output{") or line == "output":
            in_output = True
            current_io = {}
        elif line == "]" or line == "}":
            if in_input:
                inputs.append(current_io)
                in_input = False
            elif in_output:
                outputs.append(current_io)
                in_output = False
        elif in_input or in_output:
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip().strip('"')
                if k == "name":
                    current_io["name"] = v
                elif k == "data_type":
                    current_io["data_type"] = v
                elif k == "dims":
                    dims_str = v.lstrip("[").rstrip("]").strip()
                    if dims_str:
                        try:
                            current_io["dims"] = [int(x.strip()) for x in dims_str.split(",") if x.strip()]
                        except ValueError:
                            return False, "INVALID_DIMS"

    if not has_name:
        return False, "MISSING_NAME"
    if not has_platform_or_backend:
        return False, "MISSING_PLATFORM_OR_BACKEND"
    if not inputs:
        return False, "MISSING_INPUT"
    if not outputs:
        return False, "MISSING_OUTPUT"

    for io_list in (inputs, outputs):
        for io in io_list:
            if "name" not in io or not io["name"]:
                return False, "MISSING_IO_NAME"
            if "data_type" not in io or io["data_type"] not in VALID_DATA_TYPES:
                return False, "INVALID_DATA_TYPE"
            if "dims" not in io:
                return False, "MISSING_DIMS"

    return True, "OK"
