import json

SAMPLE_MANIFESTS = [
    json.dumps({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": "sha256:cfg1", "size": 500},
        "layers": [
            {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": "sha256:layerA", "size": 1024},
            {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": "sha256:layerB", "size": 2048}
        ]
    }),
    json.dumps({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": "sha256:cfg2", "size": 600},
        "layers": [
            {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": "sha256:layerB", "size": 2048},
            {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": "sha256:layerC", "size": 4096}
        ]
    }),
    json.dumps({
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": "sha256:cfg3", "size": 700},
        "layers": [
            {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": "sha256:layerA", "size": 1024},
            {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": "sha256:layerC", "size": 4096}
        ]
    })
]

TAGS_MAPPING = {
    "model:latest": SAMPLE_MANIFESTS[0],
    "model:v1": SAMPLE_MANIFESTS[0],
    "model:v2": SAMPLE_MANIFESTS[1],
    "model:alt": SAMPLE_MANIFESTS[2]
}

def parse_manifest(manifest_str):
    d = json.loads(manifest_str)
    return [
        {
            "mediaType": layer.get("mediaType"),
            "digest": layer.get("digest"),
            "size": layer.get("size")
        }
        for layer in d.get("layers", [])
    ]

def build_tag_graph(tags_dict):
    graph = {}
    for tag, m_str in tags_dict.items():
        layers = parse_manifest(m_str)
        graph[tag] = [l["digest"] for l in layers]
    return graph

def find_shared_tags(tags_dict, digest):
    graph = build_tag_graph(tags_dict)
    return sorted([tag for tag, digests in graph.items() if digest in digests])

def simulate_cp(tags_dict, src_tag, dst_tag):
    if src_tag not in tags_dict:
        raise KeyError(f"Source tag {src_tag} not found")
    return 0
