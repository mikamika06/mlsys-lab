UNSUPPORTED = {"$ref", "patternProperties", "dependencies", "not", "if", "then", "else"}

def detect_unsupported_keywords(schema):
    found = []
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k in UNSUPPORTED:
                found.append(k)
            if isinstance(v, (dict, list)):
                found.extend(detect_unsupported_keywords(v))
    elif isinstance(schema, list):
        for item in schema:
            found.extend(detect_unsupported_keywords(item))
    return sorted(list(set(found)))
