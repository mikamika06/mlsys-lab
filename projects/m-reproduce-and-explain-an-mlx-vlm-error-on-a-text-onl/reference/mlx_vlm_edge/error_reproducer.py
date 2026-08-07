def reproduce_error(model_config, prompt):
    if "vision_config" not in model_config:
        raise KeyError("Missing vision_config in text-only model configuration during multimodal token expansion.")
    return True


def explain_error():
    return "Text-only models lack vision_config and image embedding layers, causing the VLM pipeline to raise a KeyError when attempting to parse image tokens."
