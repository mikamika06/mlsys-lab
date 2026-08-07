def build_graph(manifests):
    tag_to_layers = {}
    layer_to_tags = {}
    for tag, manifest in manifests.items():
        digests = [l["digest"] for l in manifest.get("layers", [])]
        tag_to_layers[tag] = digests
        for d in digests:
            layer_to_tags.setdefault(d, []).append(tag)
    return {"tag_to_layers": tag_to_layers, "layer_to_tags": layer_to_tags}


def find_shared_tags(graph, digest):
    layer_to_tags = graph.get("layer_to_tags", {})
    return sorted(layer_to_tags.get(digest, []))
