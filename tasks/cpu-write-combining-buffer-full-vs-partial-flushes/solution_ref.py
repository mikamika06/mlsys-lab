def wc_flush_stats(addrs, line_bytes, slots):
    entries = {}
    order = []
    full_flush = 0
    partial_flush = 0

    def flush(line):
        nonlocal full_flush, partial_flush
        if len(entries[line]) == line_bytes:
            full_flush += 1
        else:
            partial_flush += 1
        del entries[line]
        order.remove(line)

    for addr in addrs:
        line = addr // line_bytes
        offset = addr % line_bytes

        if line not in entries:
            if len(entries) == slots:
                flush(order[0])
            entries[line] = set()
            order.append(line)

        entries[line].add(offset)

        if len(entries[line]) == line_bytes:
            flush(line)

    for line in list(order):
        flush(line)

    return full_flush, partial_flush
