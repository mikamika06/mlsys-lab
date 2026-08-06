def optimize_template(tokens, ts_token):
    static_part = [t for t in tokens if t != ts_token]
    return static_part + [ts_token]
