import ref


def check(workdir):
    from migration.diff import compare_relay_relax_outputs

    spec = ref.generate_spec(42)
    try:
        val = float(compare_relay_relax_outputs(spec))
    except Exception as e:
        return {"max_abs_err": 999.0, "_note": f"raised {type(e).__name__}: {str(e)[:100]}"}
    return {"max_abs_err": val}
