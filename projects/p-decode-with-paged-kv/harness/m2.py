import ref

def check(workdir):
    from pagedkv.batch import build_block_table
    m = {"table_ok": 0.0}
    seq_lens = [10, 25, 40]
    block_size = 16
    table = build_block_table(seq_lens, block_size)
    if len(table) == 3 and len(table[0]) == 1 and len(table[1]) == 2 and len(table[2]) == 3:
        m["table_ok"] = 1.0
    return m
