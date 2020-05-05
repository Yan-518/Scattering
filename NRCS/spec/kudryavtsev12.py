import numpy as np
from NRCS.spec.kudryavtsev05 import fv
from NRCS import constants as const
def c_tau(k):
    c_beta = 0.04  # wind wave growth parameter
    # wind exponent of the wave spectrum
    kmin = 3.62e+2  # rad/m
    m_sta = []
    for kk in k:
        if kk <= kmin:
            m_sta.append(1.1 * kk ** 0.742)
        else:
            m_sta.append(5.61 - 1.19 * kk + 0.118 * kk ** 2)
    m_sta = np.asarray(m_sta)
    return m_sta / (2 * c_beta)  # constant

def k_dless(k, u_10):
    return k * fv(u_10)**2/const.g

# def T(k,K):
#     nk = k.shape[0]
#     # N =
#     mk = k.reshape(nk,1)*np.gradient(np.log(N[:,0]),k[1]-k[0]).reshape(nk,1)
#     c_tau(k) * k_dless(k, u_10)**(-3/2) * mk
