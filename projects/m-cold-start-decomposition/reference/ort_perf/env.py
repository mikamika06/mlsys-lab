def check_shadow_wheel(installed_packages):
    return "onnxruntime" in installed_packages and "onnxruntime-gpu" in installed_packages
