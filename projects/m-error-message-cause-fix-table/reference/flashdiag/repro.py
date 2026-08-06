def generate_repro(config):
    script = f"import torch\nimport flash_attn\n\ndef run():\n    cfg = {config}\n    print('Running repro with', cfg)\n\nif __name__ == '__main__':\n    run()\n"
    return script
