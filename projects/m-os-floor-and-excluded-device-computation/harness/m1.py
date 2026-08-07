import ref


def check(workdir):
    from edgeexport.filtering import compute_os_floor, filter_eligible_devices

    out = {"filtering_matched": 0.0}
    ok = 0
    for var in ref.VARIANTS:
        want_floor = ref.compute_os_floor(var, ref.FEATURE_OS_MAP)
        got_floor = compute_os_floor(var, ref.FEATURE_OS_MAP)

        want_filtered = ref.filter_eligible_devices(
            ref.DEVICES, var, ref.FEATURE_OS_MAP
        )
        got_filtered = filter_eligible_devices(
            ref.DEVICES, var, ref.FEATURE_OS_MAP
        )

        if got_floor == want_floor and got_filtered == want_filtered:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"variant {var['id']}: mismatch in filtering result"

    out["filtering_matched"] = float(ok)
    return out
