def diff_package_and_compiled(pkg, compiled):
    pkg_keys = set(pkg.keys())
    comp_keys = set(compiled.keys())
    return {
        "only_in_package": sorted(list(pkg_keys - comp_keys)),
        "only_in_compiled": sorted(list(comp_keys - pkg_keys)),
        "common": sorted(list(pkg_keys & comp_keys))
    }
