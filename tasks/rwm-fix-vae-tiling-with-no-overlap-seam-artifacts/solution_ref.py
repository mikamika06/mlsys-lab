def tiled_decode(image, decode_fn, tile_size, overlap):
    H = len(image)
    W = len(image[0]) if H > 0 else 0

    def reflect_pad(arr, pad):
        h = len(arr)
        w = len(arr[0]) if h > 0 else 0
        padded = []
        for r in range(h + 2 * pad):
            orig_r = r - pad
            if orig_r < 0:
                orig_r = -orig_r
            elif orig_r >= h:
                orig_r = 2 * h - 2 - orig_r
            orig_r = max(0, min(orig_r, h - 1))

            row = []
            for c in range(w + 2 * pad):
                orig_c = c - pad
                if orig_c < 0:
                    orig_c = -orig_c
                elif orig_c >= w:
                    orig_c = 2 * w - 2 - orig_c
                orig_c = max(0, min(orig_c, w - 1))
                row.append(arr[orig_r][orig_c])
            padded.append(row)
        return padded

    padded_image = reflect_pad(image, overlap)
    out = [[0.0 for _ in range(W)] for _ in range(H)]

    for r0 in range(0, H, tile_size):
        for c0 in range(0, W, tile_size):
            r1 = min(r0 + tile_size, H)
            c1 = min(c0 + tile_size, W)

            patch_h = (r1 - r0) + 2 * overlap
            patch_w = (c1 - c0) + 2 * overlap

            patch = []
            for dr in range(patch_h):
                row_elements = []
                for dc in range(patch_w):
                    row_elements.append(padded_image[r0 + dr][c0 + dc])
                patch.append(row_elements)

            decoded = decode_fn(patch)

            patch_height = len(patch)
            patch_width = len(patch[0])
            dec_height = len(decoded)
            dec_width = len(decoded[0])

            shrink_r = (patch_height - dec_height) // 2
            shrink_c = (patch_width - dec_width) // 2

            cr = overlap - shrink_r
            cc = overlap - shrink_c

            core_h = r1 - r0
            core_w = c1 - c0

            for dr in range(core_h):
                for dc in range(core_w):
                    out[r0 + dr][c0 + dc] = decoded[cr + dr][cc + dc]

    return out
