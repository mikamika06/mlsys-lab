def find_reformats(data):
    reformats = []
    layers = data.get("layers", [])
    for i in range(1, len(layers)):
        prev = layers[i-1].get("precision")
        curr = layers[i].get("precision")
        if prev != curr or layers[i].get("is_reformat", False):
            reformats.append(layers[i].get("index"))
    return sorted(list(set(reformats)))
