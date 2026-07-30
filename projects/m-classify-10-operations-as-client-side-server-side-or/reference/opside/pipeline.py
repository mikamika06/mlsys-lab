from .classify import classify_operation


def side_sequence(config, order):
    sides = {op["name"]: classify_operation(op) for op in config["operations"]}
    return [sides[name] for name in order]


def hop_count(config, order):
    seq = side_sequence(config, order)
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def is_legal_pipeline(config, order):
    seq = side_sequence(config, order)
    forbidden = {("client", "runner"), ("runner", "client")}
    return not any((a, b) in forbidden for a, b in zip(seq, seq[1:]))
