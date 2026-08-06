def tune_max_num_seqs(latencies, slo_ttft, max_seqs_list):
    best_seqs = max_seqs_list[0]
    for seqs, lat in zip(max_seqs_list, latencies):
        if lat <= slo_ttft:
            best_seqs = seqs
        else:
            break
    return best_seqs
