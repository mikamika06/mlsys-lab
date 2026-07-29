import sys

sys.path.insert(0, ".")
from gguf_lab import read_header, read_metadata, read_tensor_info


def main(path):
    h = read_header(path)
    print(f"version {h['version']}  tensors {h['tensor_count']}  kv {h['kv_count']}")
    for k, v in read_metadata(path).items():
        print(f"  {k:34} {str(v)[:60]}")
    for t in read_tensor_info(path):
        print(f"  {t['name']:34} {t['shape']} {t['dtype']} @{t['offset']}")


if __name__ == "__main__":
    main(sys.argv[1])
