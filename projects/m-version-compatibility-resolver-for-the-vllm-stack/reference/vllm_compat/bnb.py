def detect_backend(env):
    has_cuda = env.get("HAS_CUDA", False)
    ver = env.get("BNB_VERSION", "0.0.0")
    major, minor, _ = (int(x) for x in ver.split("."))

    backend = "cuda" if has_cuda else "cpu"
    features = ["int8"]
    if major > 0 or (major == 0 and minor >= 40):
        if has_cuda:
            features.extend(["fp4", "nf4"])
    if major > 0 or (major == 0 and minor >= 42):
        features.append("optim")

    supported = has_cuda and (major > 0 or (major == 0 and minor >= 40))
    return {"backend": backend, "features": sorted(features), "supported": supported}
