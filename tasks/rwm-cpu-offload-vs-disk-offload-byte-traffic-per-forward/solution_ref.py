def offload_byte_traffic(placement, layer_sizes):
    cpu_traffic = 0
    disk_traffic = 0
    for layer, loc in placement.items():
        size = layer_sizes[layer]
        if loc == "cpu":
            cpu_traffic += size
        elif loc == "disk":
            disk_traffic += size
    return (cpu_traffic, disk_traffic)
