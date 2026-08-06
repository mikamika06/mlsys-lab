def compute_search_space_size(tiles, warps, stages):
    count = 0
    for t in tiles:
        for w in warps:
            for s in stages:
                if t[0] >= 16 and t[1] >= 16 and w >= 2 and s >= 1:
                    count += 1
    return count
