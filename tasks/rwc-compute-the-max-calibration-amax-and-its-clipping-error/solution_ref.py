import numpy as np

def calibrate_max_and_error(batches):
    all_vals = np.concatenate([b.ravel() for b in batches])
    amax = float(np.max(np.abs(all_vals)))
    scale = amax / 127.0
    recon_all = []
    for b in batches:
        q = np.round(b / scale)
        q = np.clip(q, -127, 127)
        recon = q * scale
        recon_all.append(recon.ravel())
    recon_vals = np.concatenate(recon_all)
    mse = float(np.mean((recon_vals - all_vals)**2))
    return amax, mse
