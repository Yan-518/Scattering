import numpy as np
from NRCS import constants as const
from NRCS.spec import kudryavtsev05
from NRCS.spec.kudryavtsev05 import param
from NRCS.spec.kudryavtsev05 import spec_peak
from NRCS.bistatic.Sp_bi import pol_vec_sp
from NRCS.modulation.Bragg_modulation import Trans_func

def const_eq_wb_mo(kr, K, theta_eq, eq_azi, u_10, fetch, div):
    """
    :param kp:
    :param kr:
    :param theta:
    :param azi:
    :param u_10:
    :param fetch:
    :return:
    """

    # NRCS of plumes
    wb0 = np.exp(-np.tan(theta_eq)**2/const.Swb)/(np.cos(theta_eq)**4*const.Swb)+const.yitawb/const.Swb # Kudryavstev 2003a equation (60)
    knb = min(const.br*kr, const.kwb)

    # tilting transfer function
    dtheta = theta_eq[1]-theta_eq[0]
    Mwb = np.gradient(wb0, dtheta)/wb0

    # distribution function
    phi1 = np.linspace(-np.pi, np.pi, 37)
    nk = 1024

    KK = np.linspace(10 * spec_peak(u_10, fetch), knb, nk)
    n, alpha = param(KK, u_10, fetch)

    q = np.zeros([div.shape[0], div.shape[1]])
    WB = np.zeros([div.shape[0], div.shape[1]])

    for ii in np.arange(div.shape[0]):
        for jj in np.arange(div.shape[1]):
            T = Trans_func(KK, K[ii, jj], u_10, fetch, phi1, div[ii, jj])
            Bkdir = kudryavtsev05(KK.reshape(nk, 1), u_10, fetch, phi1) * (1+abs(T.reshape(nk,1)))
            lamda = (Bkdir/alpha.reshape(nk, 1))**(n.reshape(nk, 1)+1)/(2*KK.reshape(nk, 1)) # distribution of breaking front lengths
            lamda_k = np.trapz(lamda, phi1, axis=1)
            lamda = np.trapz(lamda, KK, axis=0)
            lamda_k = np.trapz(lamda_k, KK)
            q[ii, jj] = const.cq * lamda_k
            Awb = np.trapz(np.cos(phi1 - eq_azi[jj]) * lamda, phi1) / lamda_k
            WB[ii, jj] = wb0[jj] * (1 + Mwb[jj] * const.theta_wb * Awb)
    return WB, q

def const_bi_wb(kr, K, theta_i, theta_s, theta_eq, bist_ang_az, eq_azi, u_10, fetch, div, polarization, inc_polar, re_polar, sat):
    """
    :param k:
    :param kr:
    :param theta_i:
    :param eq_azi:
    :param u_10:
    :param fetch:
    :param spec_name:
    :param polarization:
    :return:
    """
    # polarization of incident plane
    M, rot = pol_vec_sp(theta_i, theta_s, bist_ang_az, eq_azi, inc_polar, sat)
    WB, q = const_eq_wb_mo(kr, K, theta_eq, eq_azi, u_10, fetch, div)
    if polarization == 'VV':
        if re_polar == 'Bragg':
            return M * WB * np.cos(rot) ** 2, q
        else:
            return M * WB * np.sin(rot) ** 2, q
    if polarization == 'HH':
        if re_polar == 'Bragg':
            return M * WB * np.cos(rot) ** 2, q
        else:
            return M * WB * np.sin(rot) ** 2, q