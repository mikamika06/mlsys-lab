"""Trace handling utilities."""

def parse_trace(raw_data):
    return {"ops": len(raw_data.get("nodes", [])), "size": len(str(raw_data))}
