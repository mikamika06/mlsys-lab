def make_tlb_trace(pages, rounds, page_size):
    return [i * page_size for _ in range(rounds) for i in range(pages)]
