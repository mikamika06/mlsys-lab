def extract_stop_sequences(config):
    stops = set()
    if "eos_token_id" in config:
        eos = config["eos_token_id"]
        if isinstance(eos, list):
            stops.update(eos)
        else:
            stops.add(eos)
    if "stop_strings" in config:
        for s in config["stop_strings"]:
            stops.add(s)
    if "generation_config" in config:
        g = config["generation_config"]
        if "eos_token_id" in g:
            eos = g["eos_token_id"]
            if isinstance(eos, list):
                stops.update(eos)
            else:
                stops.add(eos)
        if "stop_strings" in g:
            for s in g["stop_strings"]:
                stops.add(s)
    return sorted(list(stops), key=lambda x: str(x))
