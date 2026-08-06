import ref


def check(workdir):
    from coreflow.inspector import inspect_package
    from coreflow.compatibility import resolve_availability

    pkg = ref.generate_mock_package()
    insp_res = inspect_package(pkg)

    spec = {
        "minimum_deployment_target": "iOS17",
        "operators": [{"name": "MatMul", "min_version": "iOS17"}]
    }
    compat_res = resolve_availability(spec, "iOS16")

    layout_valid = 1.0 if insp_res.get("structure_valid") else 0.0
    compat_fixed = 1.0 if compat_res.get("success") else 0.0

    return {
        "layout_valid": layout_valid,
        "compat_fixed": compat_fixed
    }
