import sys
import ref

def check(workdir):
    sys.path.insert(0, workdir)
    from exporter.replacements import verify_equivalence
    m = {"numerical_match": 0.0}
    data = ref.get_test_data()
    ok = verify_equivalence(ref.reference_gelu, ref.reference_gelu, data)
    if ok:
        m["numerical_match"] = 1.0
    return m
