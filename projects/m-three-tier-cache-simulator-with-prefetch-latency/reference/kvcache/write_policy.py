def simulate_write_traffic(operations: list[dict], bandwidth_cap_gbps: float = 10.0) -> dict:
    """Simulates write-back vs write-through policy under bandwidth constraints."""
    cap_bytes_sec = bandwidth_cap_gbps * 1e9

    def _run(mode: str):
        total_transferred = 0
        total_delay = 0.0
        vram = set()
        dram = set()

        for op in operations:
            op_type = op['type']
            key = op['key']
            sz = op['size_bytes']

            if op_type == 'write':
                vram.add(key)
                if mode == 'write_through':
                    dram.add(key)
                    total_transferred += sz
                    total_delay += sz / cap_bytes_sec
            elif op_type == 'evict':
                if key in vram:
                    vram.remove(key)
                    if mode == 'write_back':
                        dram.add(key)
                        total_transferred += sz
                        total_delay += sz / cap_bytes_sec

        return {
            'total_bytes_transferred': total_transferred,
            'total_delay_sec': total_delay
        }

    return {
        'write_through': _run('write_through'),
        'write_back': _run('write_back')
    }
