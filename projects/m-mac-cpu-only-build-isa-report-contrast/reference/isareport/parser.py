def parse_build_log(log_str):
    features = {}
    for line in log_str.strip().split("\n"):
        if "Detected" in line:
            parts = line.split(":")
            if len(parts) == 2:
                k = parts[0].replace("Detected", "").strip().lower()
                v = parts[1].strip() == "True"
                features[k] = v
        elif "CMAKE_ARGS" in line:
            if "GGML_NATIVE=ON" in line:
                features["native_flag"] = True
            else:
                features["native_flag"] = False
    return features
