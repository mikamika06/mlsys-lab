import sys
sys.path.insert(0, ".")
from blobgraph.graph import build_tag_graph, find_shared_tags
from blobgraph.cp import simulate_cp

SAMPLE_TAGS = {
    "test:latest": '{"layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": "sha256:abc", "size": 100}]}'
}

def test_graph_returns_non_empty_layers():
    g = build_tag_graph(SAMPLE_TAGS)
    assert "test:latest" in g
    assert len(g["test:latest"]) > 0
    assert g["test:latest"][0] == "sha256:abc"

def test_find_shared_tags_detects_owner():
    tags = {
        "t1": '{"layers": [{"digest": "sha256:shared", "size": 10}]}',
        "t2": '{"layers": [{"digest": "sha256:shared", "size": 10}]}'
    }
    shared = find_shared_tags(tags, "sha256:shared")
    assert sorted(shared) == ["t1", "t2"]

def test_simulate_cp_costs_zero():
    cost = simulate_cp(SAMPLE_TAGS, "test:latest", "test:copy")
    assert cost == 0
