import json


def rename_io(spec, input_map, output_map):
    res = json.loads(json.dumps(spec))
    if "inputs" in res:
        for item in res["inputs"]:
            if item.get("name") in input_map:
                item["name"] = input_map[item["name"]]
    if "outputs" in res:
        for item in res["outputs"]:
            if item.get("name") in output_map:
                item["name"] = output_map[item["name"]]
    if "nodes" in res:
        for node in res["nodes"]:
            if "inputs" in node:
                node["inputs"] = [input_map.get(i, i) for i in node["inputs"]]
            if "outputs" in node:
                node["outputs"] = [output_map.get(o, o) for o in node["outputs"]]
    return res
