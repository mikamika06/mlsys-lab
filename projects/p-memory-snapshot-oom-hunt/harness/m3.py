import ref


def check(workdir):
    from leak.detector import MemorySnapshotAnalyzer

    m = {"ref_chain_found": 0.0}
    data = {
        "objects": {
            "obj_a": {"parent": None},
            "obj_b": {"parent": "obj_a"},
            "obj_c": {"parent": "obj_b"}
        }
    }
    analyzer = MemorySnapshotAnalyzer(data)
    try:
        chain = analyzer.find_reference_chain("obj_c")
        if chain == ["obj_a", "obj_b", "obj_c"]:
            m["ref_chain_found"] = 1.0
    except Exception:
        pass
    return m
