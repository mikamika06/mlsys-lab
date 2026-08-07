import ref

def get_deploy_script() -> str:
    return "python -m mlx_lm.serve --model model_quantized"
