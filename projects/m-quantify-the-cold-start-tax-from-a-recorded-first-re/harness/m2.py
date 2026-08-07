import ref


def check(workdir):
    from coldstart.tax import compute_cold_start_tax
    from coldstart.scale import fraction_exposed

    trace_data = ref.generate_trace()
    timeout = 3.0

    want_tax = ref.compute_cold_start_tax(trace_data, timeout)
    got_tax = compute_cold_start_tax(trace_data, timeout)

    want_frac = ref.fraction_exposed(trace_data, timeout)
    got_frac = fraction_exposed(trace_data, timeout)

    tax_err = abs(got_tax - want_tax) / (abs(want_tax) + 1e-9)
    frac_err = abs(got_frac - want_frac) / (abs(want_frac) + 1e-9)

    out = {
        "tax_rel_err": float(tax_err),
        "exposure_rel_err": float(frac_err)
    }
    return out
