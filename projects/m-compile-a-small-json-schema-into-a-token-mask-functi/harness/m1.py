import ref


def check(workdir):
    import sys

    sys.path.insert(0, workdir)
    from schema_runner.compiler import SchemaMaskCompiler

    compiler = SchemaMaskCompiler(ref.VOCAB, ref.EOS_ID)
    matched = 0

    for i, schema in enumerate(ref.SCHEMAS):
        mask_fn = compiler.compile(schema)
        if callable(mask_fn):
            allowed_initial = mask_fn([])
            if isinstance(allowed_initial, set) and len(allowed_initial) > 0:
                matched += 1

    out = {
        "schemas_matched": float(matched),
        "total_schemas": float(len(ref.SCHEMAS)),
    }
    return out
