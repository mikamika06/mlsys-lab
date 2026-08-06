def build_convert_args(config):
    args = ["--model", config["model"]]
    if config["quantize"]:
        args.extend(["-q", "--q-bits", str(config["bits"])])
        if config.get("group_size", 0) > 0:
            args.extend(["--q-group-size", str(config["group_size"])])
    return args
