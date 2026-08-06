import ref

def check(workdir):
    from latency.analyzer import find_losing_batch_size
    tables = ref.generate_fixtures()
    ok = 0
    total = len(tables)
    slo = 30.0
    for table in tables:
        expected = None
        for row in table:
            if row["latency"] > slo:
                expected = row["batch_size"]
                break
        if expected is None and table:
            expected = table[-1]["batch_size"]

        got = find_losing_batch_size(table, slo)
        if got == expected:
            ok += 1

    match = 1.0 if ok == total else 0.0
    return {"argmin_index": match}
