def build_and_validate_wiring(cfg):
    return {
        "inputs": cfg["step1"]["inputs"],
        "outputs": cfg["step2"]["outputs"],
        "steps": [
            {"name": cfg["step1"]["name"], "input_map": {i: i for i in cfg["step1"]["inputs"]}, "output_map": {o: o for o in cfg["step1"]["outputs"]}},
            {"name": cfg["step2"]["name"], "input_map": {i: i for i in cfg["step2"]["inputs"]}, "output_map": {o: o for o in cfg["step2"]["outputs"]}}
        ]
    }
