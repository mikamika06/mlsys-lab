from blobgraph.manifest import parse_manifest

def build_tag_graph(tags_dict):
    graph = {}
    for tag, m_str in tags_dict.items():
        layers = parse_manifest(m_str)
        graph[tag] = [l["digest"] for l in layers]
    return graph

def find_shared_tags(tags_dict, digest):
    graph = build_tag_graph(tags_dict)
    return sorted([tag for tag, digests in graph.items() if digest in digests])
