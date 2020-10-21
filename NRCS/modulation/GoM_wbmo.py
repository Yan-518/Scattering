import numpy as np
from NRCS import constants as const
from NRCS.spec import kudryavtsev05
from NRCS.spec.kudryavtsev05 import param
from NRCS.spec.kudryavtsev05 import spec_peak
from NRCS.modulation.Bragg_modulation import Trans_func

def GoM_wbmo(kr, K, theta, azimuth, u_10, fetch, wind_dir, div, polarization):
    """
    :param kp:
    :param kr:
    :param theta:
    :param azi:
    :param u_10:
    :param fetch:
    :return:
    """
    if polarization == 'VH':
        print('no modulation cross-pol')
        return

    nphi = theta.shape[0]

    ind = np.where(np.degrees(azimuth) == wind_dir)[0]

    # NRCS of plumes
    wb0 = np.exp(-np.tan(theta)**2/const.Swb)/(np.cos(theta)**4*const.Swb)+const.yitawb/const.Swb # Kudryavstev 2003a equation (60)
    knb = min(const.br*kr, const.kwb)

    # tilting transfer function
    dtheta = theta[1]-theta[0]
    Mwb = np.gradient(wb0, dtheta)/wb0

    # distribution function
    # in radians azimuth of breaking surface area: -pi/2,pi/2
    # phi1 = np.linspace(-np.pi/2, np.pi/2, nphi)
    phi1 = np.linspace(-np.pi, np.pi, nphi)
    nk = 1024

    q = np.zeros([div.shape[0],div.shape[1]])
    WB = np.zeros([div.shape[0],div.shape[1]])

    for ii in np.arange(div.shape[0]):
        for jj in np.arange(div.shape[1]):
            KK = np.linspace(10 * spec_peak(u_10[ii,jj], fetch), knb, nk)
            T = Trans_func(KK, K[ii, jj], u_10[ii, jj], fetch, azimuth, div[ii, jj])
            Bkdir = kudryavtsev05(KK.reshape(nk, 1), u_10[ii, jj], fetch, phi1)* (1+abs(T.reshape(nk,1)))
            n, alpha = param(KK, u_10[ii, jj], fetch)
            lamda = (Bkdir/alpha.reshape(nk, 1))**(n.reshape(nk, 1)+1)/(2*KK.reshape(nk, 1)) # distribution of breaking front lengths
            lamda_k = np.trapz(lamda, phi1, axis=1)
            lamda = np.trapz(lamda, KK, axis=0)
            lamda_k = np.trapz(lamda_k, KK)
            q[ii, jj] = const.cq * lamda_k
            Awb = np.trapz(np.cos(phi1 - azimuth[ind]) * lamda, phi1) / lamda_k
            WB[ii, jj] = wb0[jj]*(1+Mwb[jj]*const.theta_wb*Awb)
    return WB, q
