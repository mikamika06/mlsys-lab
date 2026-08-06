def map_tensors(raw_names):
    out = {}
    for name in raw_names:
        if name == "vision_model.embeddings.patch_embedding.weight":
            out[name] = "v.patch_embd.weight"
        elif name == "vision_model.embeddings.class_embedding":
            out[name] = "v.class_embd"
        elif name == "vision_model.embeddings.position_embedding.weight":
            out[name] = "v.pe.weight"
        elif name.startswith("vision_model.post_layernorm."):
            out[name] = name.replace("vision_model.post_layernorm.", "v.post_ln.")
        elif name.startswith("multi_modal_projector.linear_"):
            parts = name.split(".")
            out[name] = f"mm.proj.{parts[1].split('_')[1]}.{parts[2]}"
        elif name.startswith("vision_model.encoder.layers."):
            parts = name.split(".")
            idx = parts[3]
            sub = parts[4]
            last = parts[-2]
            prop = parts[-1]
            if sub == "self_attn":
                if last == "out_proj":
                    out[name] = f"v.blk.{idx}.attn_out.{prop}"
                else:
                    out[name] = f"v.blk.{idx}.attn_{last[0]}.{prop}"
            elif sub == "layer_norm1":
                out[name] = f"v.blk.{idx}.ln1.{prop}"
            elif sub == "layer_norm2":
                out[name] = f"v.blk.{idx}.ln2.{prop}"
            elif sub == "mlp":
                if last == "fc1":
                    out[name] = f"v.blk.{idx}.ffn_down.{prop}"
                elif last == "fc2":
                    out[name] = f"v.blk.{idx}.ffn_up.{prop}"
    return out
