"""Register tile shape reconstruction."""


def derive_register_tile(num_registers, vector_len_bytes, elem_size_bytes):
    vec_elems = vector_len_bytes // elem_size_bytes
    reserved_for_b = 1
    reserved_for_a = 1
    avail = num_registers - (reserved_for_a + reserved_for_b)
    if avail < 1:
        return (1, vec_elems)

    best_mr = 1
    best_nr_vec = 1
    best_product = 0

    max_mr = avail
    for mr in range(1, max_mr + 1):
        for nr_vec in range(1, avail // mr + 1):
            if mr * nr_vec <= avail:
                prod = mr * nr_vec
                if prod > best_product or (
                    prod == best_product and abs(mr - nr_vec) < abs(best_mr - best_nr_vec)
                ):
                    best_product = prod
                    best_mr = mr
                    best_nr_vec = nr_vec

    return (best_mr, best_nr_vec * vec_elems)
