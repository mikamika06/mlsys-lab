class ModelConverter:
    def __init__(self, config: dict, state_dict: dict):
        self.config = config
        self.state_dict = state_dict

    def get_metadata(self) -> dict:
        raise NotImplementedError

    def get_tensors(self) -> dict:
        raise NotImplementedError


class MiniGPTConverter(ModelConverter):
    def get_metadata(self) -> dict:
        return {
            "general.architecture": "minigpt",
            "minigpt.block_count": self.config.get("n_layers", 0),
            "minigpt.embedding_length": self.config.get("n_embd", 0),
            "minigpt.attention.head_count": self.config.get("n_head", 0),
        }

    def get_tensors(self) -> dict:
        out = {}
        for k, v in self.state_dict.items():
            if k == "minigpt.embed.weight":
                out["token_embd.weight"] = v
            elif k == "minigpt.output.weight":
                out["output.weight"] = v
            elif k.startswith("minigpt.layers."):
                parts = k.split(".")
                layer_idx = parts[2]
                sub = parts[3]
                proj = parts[4]
                if sub == "attn":
                    if proj == "q_proj":
                        out[f"blk.{layer_idx}.attn_q.weight"] = v
                    elif proj == "k_proj":
                        out[f"blk.{layer_idx}.attn_k.weight"] = v
                    elif proj == "v_proj":
                        out[f"blk.{layer_idx}.attn_v.weight"] = v
                elif sub == "mlp":
                    if proj == "up_proj":
                        out[f"blk.{layer_idx}.ffn_up.weight"] = v
                    elif proj == "down_proj":
                        out[f"blk.{layer_idx}.ffn_down.weight"] = v
        return out
