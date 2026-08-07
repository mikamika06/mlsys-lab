def check_overlap_feasibility(comm_time, compute_time, overlap_efficiency):
    effective_comm = comm_time * (1.0 - overlap_efficiency)
    return effective_comm <= compute_time
