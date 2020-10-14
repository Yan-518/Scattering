import numpy as np
import NRCS.constants as const
from NRCS.spec.kudryavtsev05 import fv
from NRCS.model import Wave_breaking
from NRCS.modulation.NRCS_contrasts import Wave_breaking_new

def WB_emp(K, kr, u_10, tsc):
    # empirical wave breaking contrasts from Eq.17 in [Kudryavtsev, 2012]

    cq = 470
    U = fv(u_10)
    kb = kr / 10
    omegab = np.sqrt(const.g * kb + const.gamma * kb ** 3/const.rho_water)
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)
    wb_c = - cq * np.log(U * kb / np.sqrt(const.g * K)) * const.g * divergence / (U**2 * kb * omegab)
    return wb_c

def WB_contrasts(k, K, kr, theta, azimuth, u_10, fetch, spec_name, tsc, polarization):
    nphi = theta.shape[0]
    nazi = azimuth.shape[0]
    WB_old = np.zeros([k.shape[0], k.shape[1], nazi, nphi])
    q_old = np.zeros([k.shape[0], k.shape[1]])
    for ii in np.arange(k.shape[0]):
        for jj in np.arange(k.shape[1]):
            WB_old[ii, jj, :, :], q_old[ii, jj] = Wave_breaking(kr, theta, azimuth, u_10[ii, jj], fetch, spec_name, polarization)
    WB_new, q_new = Wave_breaking_new(K, kr, theta, azimuth, u_10, fetch, spec_name, tsc)
    return (q_new - q_old) / q_old
