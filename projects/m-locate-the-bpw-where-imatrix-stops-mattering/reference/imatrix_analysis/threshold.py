import numpy as np


def find_imatrix_convergence_bpw(tensors_data, tol=1e-3):
    convergence_bpws = []
    for t in tensors_data:
        bpws = np.array(t["bpws"], dtype=np.float64)
        u_err = np.array(t["unweighted_errors"], dtype=np.float64)
        i_err = np.array(t["imatrix_errors"], dtype=np.float64)

        diff = np.abs(u_err - i_err)
        below_tol = np.where(diff <= tol)[0]

        if len(below_tol) > 0:
            idx = int(below_tol[0])
            convergence_bpws.append(float(bpws[idx]))
        else:
            convergence_bpws.append(float(bpws[-1]))

    return float(np.mean(convergence_bpws))
