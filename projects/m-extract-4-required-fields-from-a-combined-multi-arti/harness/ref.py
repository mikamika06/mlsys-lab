import os
import tempfile
import shutil

REQUIRED_KEYS = ["graph_id", "node_count", "op_name", "compile_status"]

def generate_fixtures():
    log_content = (
        "[TORCH_LOGS]: graph_id=42 node_count=15 op_name=aten::add compile_status=SUCCESS\n"
        "[TORCH_LOGS]: other_field=ignored graph_id=99 node_count=3 op_name=aten::mul compile_status=FAILED\n"
    )
    debug_files = ["output.code", "compiled_subgraph.py", "ir.txt", "config.json"]
    return log_content, debug_files

def extract_fields(log_text):
    results = []
    for line in log_text.splitlines():
        if "[TORCH_LOGS]" in line:
            parts = line.replace("[TORCH_LOGS]:", "").strip().split()
            item = {}
            for p in parts:
                if "=" in p:
                    k, v = p.split("=", 1)
                    if k in REQUIRED_KEYS:
                        item[k] = v
            if all(k in item for k in REQUIRED_KEYS):
                results.append(item)
    return results

def verify_signature(func_code, expected_sig):
    return expected_sig in func_code

def enumerate_debug_dir(dir_path):
    if not os.path.isdir(dir_path):
        return []
    files = []
    for root, _, filenames in os.walk(dir_path):
        for f in filenames:
            files.append(f)
    return sorted(files)
