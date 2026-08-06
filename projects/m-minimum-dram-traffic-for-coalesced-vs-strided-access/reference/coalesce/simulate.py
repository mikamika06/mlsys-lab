def simulate_warp_coalescing(addresses, element_size=4):
    sectors = set()
    for addr in addresses:
        start_sector = addr // 32
        end_sector = (addr + element_size - 1) // 32
        for s in range(start_sector, end_sector + 1):
            sectors.add(s)
    return len(sectors)
