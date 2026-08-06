from convertkit.names import map_index
from convertkit.plan import conversion_plan
from convertkit.shapes import check_attention


def audit(tensors, meta, prefix, experts=0):
    warnings = []

    mapping = map_index(tensors, experts)
    for name in mapping["unmapped"]:
        if name.endswith("_exps.weight"):
            warnings.append(
                "%s holds every expert fused together and no expert count was "
                "given; converting it as one tensor drops all but one expert"
                % name)
        else:
            warnings.append("%s has no counterpart in the target naming" % name)

    plan = conversion_plan(tensors, experts)
    if plan["expansion"] > 2.0:
        warnings.append(
            "output would grow %.2fx to %.1f GB: the source is quantised and "
            "the target dtype is not"
            % (plan["expansion"], plan["write_bytes"] / 1e9))

    try:
        problems = check_attention(tensors, meta, prefix)["problems"]
    except KeyError:
        problems = []
    for p in problems:
        warnings.append("attention shapes disagree with the metadata: %s" % p)
    return warnings
