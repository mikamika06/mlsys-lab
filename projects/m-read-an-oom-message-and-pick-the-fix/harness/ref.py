SAMPLE_OOMS = [
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.20 GiB (GPU 0; 23.69 GiB total capacity; 14.50 GiB already allocated; 1.05 GiB free; 15.55 GiB reserved in total by PyTorch). If reserved memory is significantly larger than allocated memory try setting max_split_size_mb to avoid fragmentation.",
    "torch.cuda.OutOfMemoryError: Tried to allocate 512.00 MiB (GPU 0; 16.00 GiB total capacity; 15.50 GiB already allocated; 50.00 MiB free; 15.80 GiB reserved)",
]

SAMPLE_TRACES = [
    [(1, 1024), (2, 512), (-1, 0), (3, 2048), (-2, 0), (4, 256)]
]

CANDIDATE_SIZES = [16, 32, 64, 128]
