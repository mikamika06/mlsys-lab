A teammate is trying to fit a small KV cache into memory for an edge device. "I palettized the attention weights to 4 bits using k-means, but the total memory barely went down!"

You suspect they forgot that k-means palettization means you have to store the palette itself (the cluster centroids). While 4-bit scalar palettization shrinks the indices to just 4 bits per element, you still have a $16 \times 4$ byte lookup table. If they tried 8-bit vector palettization with block size 2, the table is $256 \times 2 \times 4 = 2048$ bytes. For small tensors, the table size can completely eliminate the compression gains!

Your task:
1. Implement scalar k-means palettization in `palettize_scalar`, and a helper `palettize_size_bytes` that calculates the *exact* byte count for the compressed tensor. Note: float32 palettes cost 4 bytes per scalar, and bit-packed indices round up to the nearest byte.
2. Implement `palettize_vector` to group elements into blocks and palettize them together. This lets you compare identical bitrates (e.g. 4-bit scalar vs 8-bit block-of-2) to see which preserves the tensor better.
3. Write a regression test to ensure future team members don't accidentally ignore the palette size in their byte accounting.

Use `np.linspace(tensor.min(), tensor.max(), K)` for scalar initialization, and `vecs[np.linspace(0, vecs.shape[0]-1, K).astype(int)]` for vector initialization. If a cluster has no assigned points during an iteration, leave its centroid unchanged.
