import hashlib
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


def detect_duplicate_blobs(blobs):
    seen = {}
    duplicates = set()
    for path, data in sorted(blobs.items()):
        h = hashlib.sha256(data).hexdigest()
        if h in seen:
            duplicates.add(path)
            duplicates.add(seen[h])
        else:
            seen[h] = path
    return sorted(list(duplicates))


def diff_package_and_compiled(pkg, compiled):
    pkg_keys = set(pkg.keys())
    comp_keys = set(compiled.keys())
    return {
        "only_in_package": sorted(list(pkg_keys - comp_keys)),
        "only_in_compiled": sorted(list(comp_keys - pkg_keys)),
        "common": sorted(list(pkg_keys & comp_keys))
    }


SPECS = [
    (
        {
            "inputs": [{"name": "data"}],
            "outputs": [{"name": "prob"}],
            "nodes": [{"inputs": ["data"], "outputs": ["prob"]}]
        },
        {"data": "input_tensor"},
        {"prob": "output_logits"}
    )
]


BLOBS = [
    {
        "weights/layer1.bin": b"abc",
        "weights/layer2.bin": b"xyz",
        "weights/layer3.bin": b"abc"
    }
]
