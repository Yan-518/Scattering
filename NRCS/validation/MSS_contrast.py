import numpy as np
from NRCS import spec

def MSS_contrast(k, kp, kc, u_10, fetch, azimuth, T, spec_name):
    nk = k.shape[0]
    specf = spec.models[spec_name]
    if spec_name == 'elfouhaily':
        B = specf(k.reshape(nk, 1), u_10, fetch) * k.reshape(nk, 1) ** 3
    else:
        B = specf(k.reshape(nk, 1), u_10, fetch, azimuth)
        B = np.trapz(B, azimuth, axis=1)

    s2 = B.reshape(nk, 1) / k.reshape(nk, 1)
    s2 = np.trapz(s2[kc <= k][k[kc <= k] <= kp], k[kc <= k][k[kc <= k] <= kp])
    ss = B[:, 0] * np.abs(T) / k[:, 0]
    ss = ss[:, :, (kc <= k)[:, 0]]
    ss = ss[:, :, (k[kc <= k] <= kp)]
    ss = np.trapz(ss, k[kc <= k][k[kc <= k] <= kp], axis=2)
    return ss / s2
