import numpy as np

class FSDPCheckpointManager:
    def __init__(self, checkpoint_data):
        self.data = checkpoint_data

    def parse_structure(self):
        shards = self.data["shards"]
        metadata = []
        for s in shards:
            metadata.append({
                "rank": s["rank"],
                "world_size": s["world_size"],
                "shape": list(s["tensor"].shape),
                "global_shape": s["global_shape"],
                "sharded_dim": s["sharded_dim"]
            })
        return {"num_shards": len(shards), "metadata": metadata}

    def map_shards(self):
        parsed = self.parse_structure()
        mapping = {}
        for meta in parsed["metadata"]:
            rank = meta["rank"]
            mapping[rank] = {
                "tensor_shape": meta["shape"],
                "global_shape": meta["global_shape"],
                "sharded_dim": meta["sharded_dim"]
            }
        return mapping

    def convert_to_unified(self):
        shards = self.data["shards"]
        global_shape = shards[0]["global_shape"]
        sharded_dim = shards[0]["sharded_dim"]
        sorted_shards = sorted(shards, key=lambda x: x["rank"])
        tensors = [s["tensor"] for s in sorted_shards]
        unified_tensor = np.concatenate(tensors, axis=sharded_dim)
        return {"weight": unified_tensor, "global_shape": global_shape}

    def load_on_cards(self, target_world_size):
        unified = self.convert_to_unified()
        full_weight = unified["weight"]
        global_shape = unified["global_shape"]
        shard_size = global_shape[0] // target_world_size
        new_shards = []
        for rank in range(target_world_size):
            start = rank * shard_size
            end = (rank + 1) * shard_size
            new_shards.append({
                "rank": rank,
                "world_size": target_world_size,
                "tensor": full_weight[start:end, :].copy(),
                "sharded_dim": 0,
                "global_shape": global_shape
            })
        return {"world_size": target_world_size, "shards": new_shards}

    def verify_loss(self, target_world_size, test_input):
        original_weight = self.data["full_weight"]
        orig_loss = float(np.sum((test_input @ original_weight) ** 2))

        loaded_ckpt = self.load_on_cards(target_world_size)
        tensors = [s["tensor"] for s in sorted(loaded_ckpt["shards"], key=lambda x: x["rank"])]
        reconstructed = np.concatenate(tensors, axis=0)
        new_loss = float(np.sum((test_input @ reconstructed) ** 2))

        return abs(orig_loss - new_loss) < 1e-5
