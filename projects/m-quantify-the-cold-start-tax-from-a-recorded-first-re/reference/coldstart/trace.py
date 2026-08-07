def parse_trace(records):
    return sorted(records, key=lambda x: x["arrival"])
