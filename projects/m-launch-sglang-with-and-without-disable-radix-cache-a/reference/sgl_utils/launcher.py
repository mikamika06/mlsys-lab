def build_launch_command(model_path, port=30000, disable_radix_cache=False, extra_args=None):
    cmd = ["python", "-m", "sglang.launch_server", "--model-path", str(model_path), "--port", str(port)]
    if disable_radix_cache:
        cmd.append("--disable-radix-cache")
    if extra_args:
        for k, v in extra_args.items():
            cmd.extend([str(k), str(v)])
    return cmd
