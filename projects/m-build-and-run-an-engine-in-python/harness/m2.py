import ref

def check(workdir):
    from engine.builder import trtexec_to_config, build_engine
    ok = 0
    total = len(ref.TRTEXEC_ARGS)
    for args in ref.TRTEXEC_ARGS:
        got_cfg = trtexec_to_config(args)
        want_cfg = {
            "fp16": args.get("fp16", False),
            "int8": args.get("int8", False),
            "max_workspace_size": args.get("memPoolSize", 1 << 30),
            "max_batch_size": args.get("batch", 1),
        }
        if got_cfg == want_cfg:
            engine = build_engine("dummy_net", got_cfg)
            if engine and engine.get("serialized"):
                ok += 1
    passed = 1.0 if ok == total else 0.0
    return {"builder_matched": passed}
