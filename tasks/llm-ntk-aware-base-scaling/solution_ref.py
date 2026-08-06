import math

def scale_rope_base(theta: list[float] | float,
                    factor: list[float] | float) -> list[float] | float:
    """
    NTK‑aware RoPE base scaling.

    Parameters
    ----------
    theta : list of float or scalar
        Original RoPE base(s).
    factor : list of float or scalar
        Scaling factor(s).  The new base is computed as ``theta ** factor``.
    Returns
    -------
    list of float or float
        Scaled base(s).
    """
    is_scalar_theta = isinstance(theta, (int, float))
    is_scalar_factor = isinstance(factor, (int, float))

    theta_list = [float(theta)] if is_scalar_theta else [float(x) for x in theta]
    factor_list = [float(factor)] if is_scalar_factor else [float(x) for x in factor]

    if len(theta_list) == 1 and len(factor_list) > 1:
        theta_list = theta_list * len(factor_list)
    elif len(factor_list) == 1 and len(theta_list) > 1:
        factor_list = factor_list * len(theta_list)

    result = []
    for t, f in zip(theta_list, factor_list):
        result.append(t ** f)

    if is_scalar_theta and is_scalar_factor:
        return result[0]
    return result
