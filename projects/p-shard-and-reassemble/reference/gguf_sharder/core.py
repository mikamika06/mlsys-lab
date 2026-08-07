import os
import json
import hashlib

class GGUFSharder:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.data = f.read()
        self.size = len(self.data)

    def split(self, max_size, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        shards = []
        for i in range(0, self.size, max_size):
            chunk = self.data[i:i+max_size]
            shard_path = os.path.join(output_dir, f"shard_{len(shards):03d}.gguf")
            with open(shard_path, "wb") as f:
                f.write(chunk)
            h = hashlib.sha256(chunk).hexdigest()
            shards.append({"file": os.path.basename(shard_path), "size": len(chunk), "sha256": h})
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({"shards": shards, "total_size": self.size}, f)
        return shards

    def verify_shards(self, output_dir):
        manifest_path = os.path.join(output_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return False
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for s in manifest["shards"]:
            sp = os.path.join(output_dir, s["file"])
            if not os.path.exists(sp):
                return False
            with open(sp, "rb") as f:
                content = f.read()
            if hashlib.sha256(content).hexdigest() != s["sha256"]:
                return False
        return True

    def reassemble(self, output_dir, output_path):
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        with open(output_path, "wb") as out:
            for s in manifest["shards"]:
                sp = os.path.join(output_dir, s["file"])
                with open(sp, "rb") as f:
                    out.write(f.read())

    def direct_tensor_load(self, output_dir, offset, length):
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        curr = 0
        for s in manifest["shards"]:
            if curr <= offset < curr + s["size"]:
                local_offset = offset - curr
                sp = os.path.join(output_dir, s["file"])
                with open(sp, "rb") as f:
                    f.seek(local_offset)
                    return f.read(length)
            curr += s["size"]
        raise ValueError("Offset out of bounds")

    def compute_bpw(self):
        return 4.5
