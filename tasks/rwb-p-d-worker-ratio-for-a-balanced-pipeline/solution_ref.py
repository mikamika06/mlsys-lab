from fractions import Fraction


def balanced_pd_ratio(prefill_tps_per_worker, decode_tps_per_worker, mean_input_len, mean_output_len):
    ratio = Fraction(
        decode_tps_per_worker * mean_input_len,
        prefill_tps_per_worker * mean_output_len,
    )
    return ratio.numerator, ratio.denominator
