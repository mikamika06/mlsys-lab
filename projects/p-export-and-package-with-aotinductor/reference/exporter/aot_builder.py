import os
from typing import Any


def compile_aot_artifact(exported_program: Any, output_so_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_so_path)), exist_ok=True)
    with open(output_so_path, "wb") as f:
        f.write(b"\x7fELF_FAKE_AOT_INDUCTOR_SO_BINARY_DATA")
    return output_so_path
