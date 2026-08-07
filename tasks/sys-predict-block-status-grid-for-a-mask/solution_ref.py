def block_status_grid(mask: list[list[bool]], tile_h: int, tile_w: int) -> list[list[int]]:
    H = len(mask)
    W = len(mask[0])
    n_h = H // tile_h
    n_w = W // tile_w
    area = tile_h * tile_w
    status = []
    for i in range(n_h):
        row = []
        for j in range(n_w):
            tile_sum = 0
            for r in range(tile_h):
                for c in range(tile_w):
                    tile_sum += int(mask[i * tile_h + r][j * tile_w + c])
            if tile_sum == 0:
                row.append(2)
            elif tile_sum == area:
                row.append(0)
            else:
                row.append(1)
        status.append(row)
    return status
