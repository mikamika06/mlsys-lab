def min_dram_traffic(num_elements, element_size, stride):
    if num_elements <= 0:
        return 0
    sectors = set()
    for i in range(num_elements):
        addr = i * stride * element_size
        start_sector = addr // 32
        end_sector = (addr + element_size - 1) // 32
        for s in range(start_sector, end_sector + 1):
            sectors.add(s)
    return len(sectors) * 32
