import numpy as np
from NRCS.modulation import const_spec

def const_mss(k, K, u_10, fetch, azimuth, div, wind_dir):
    B_old, B_new = const_spec(k, K, u_10, fetch, azimuth, div, wind_dir)
    s2_new = np.trapz(B_new / k, k, axis=2)
    s2_old = np.trapz(B_old / k, k, axis=2)
    return (s2_new - s2_old) / s2_old