import numpy as np
import NRCS.constants as const
from NRCS.spec.kudryavtsev05 import fv
from NRCS.modulation import GoM_spec

def GoM_MSS_em(K, u_10, div):
    # Empirical mss from Eq.17 in [Kudryavtsev, 2012]
    cs = 180
    U = fv(u_10)
    return -cs * div/ (U * np.sqrt(const.ky * K))

def GoM_mss(k, K, u_10, fetch, azimuth, div, wind_dir):
    B_old, B_new = GoM_spec(k, K, u_10, fetch, azimuth, div, wind_dir)
    s2_new = np.trapz(B_new / k, k, axis=2)
    s2_old = np.trapz(B_old / k, k, axis=2)
    return (s2_new - s2_old) / s2_old