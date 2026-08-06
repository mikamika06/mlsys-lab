import ref

def check(workdir):
    from bertfuse.fusion import fuse_bert_graph
    ok = 0
    for g in ref.GRAPHS:
        want = ref.fuse_bert_graph(g)
        got = fuse_bert_graph(g)
        if got == want:
            ok += 1
    return {"fusions_matched": 1.0 if ok == len(ref.GRAPHS) else 0.0}
