def check_tile_alignment(block_m, block_n, block_k):
    if block_m % 16 != 0 or block_n % 16 != 0 or block_k % 16 != 0:
        return False
    return True
