import os
import ref

def check(workdir):
    from ggufwriter.writer import GGUFWriter
    out = {"states_passed": 0.0}
    test_path = os.path.join(workdir, "test_out.gguf")
    try:
        writer = GGUFWriter(test_path)
        writer.add_header({"version": 1})
        writer.add_tensor("weight_a", b"\x00" * 16)
        writer.add_tensor("weight_b", b"\x01" * 16)
        writer.write_header_to_file()
        if os.path.exists(test_path):
            out["states_passed"] = 3.0
    except Exception as e:
        out["_note"] = f"Failed state machine test: {e}"
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)
    return out
