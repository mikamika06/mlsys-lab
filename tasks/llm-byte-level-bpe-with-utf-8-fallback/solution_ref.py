def byte_bpe_encode(text, vocab, merges):
    symbols = [bytes([b]) for b in text.encode("utf-8")]

    while True:
        best = None
        best_rank = None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            if pair in merges:
                rank = merges[pair]
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best = pair

        if best is None:
            break

        merged = best[0] + best[1]
        result = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == best[0] and symbols[i + 1] == best[1]:
                result.append(merged)
                i += 2
            else:
                result.append(symbols[i])
                i += 1
        symbols = result

    return [vocab[symbol] for symbol in symbols]
