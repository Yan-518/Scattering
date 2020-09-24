import numpy as np
from NRCS import constants as const
from NRCS import spec
from NRCS import spread
from NRCS.spec.kudryavtsev05 import param
from NRCS.spec.kudryavtsev05 import spec_peak
from stereoid.oceans.bistatic_pol import elfouhaily

def eq_wb(kr, theta_eq, eq_azi, u_10, fetch, spec_name):
    """
    :param kp:
    :param kr:
    :param theta:
    :param azi:
    :param u_10:
    :param fetch:
    :return:
    """

    nphi = theta_eq.shape[0]

    # Spectral model
    # Omni directional spectrum model name
    specf = spec.models[spec_name]

    # NRCS of plumes
    wb0 = np.exp(-np.tan(theta_eq)**2/const.Swb)/(np.cos(theta_eq)**4*const.Swb)+const.yitawb/const.Swb # Kudryavstev 2003a equation (60)
    knb = min(const.br*kr, const.kwb)

    # tilting transfer function
    dtheta = theta_eq[1]-theta_eq[0]
    Mwb = np.gradient(wb0, dtheta)/wb0

    # distribution function
    phi1 = np.linspace(-np.pi, np.pi, nphi) # in radians azimuth of breaking surface area: -pi/2,pi/2
    nk = 1024
    K = np.linspace(10 * spec_peak(u_10, fetch), knb, nk)
    # K = np.linspace(spec_peak(u_10, fetch), knb, nk)

    if spec_name == 'elfouhaily':
        # Directional spectrum model name
        spreadf = spread.models[spec_name]
        Bkdir = specf(K.reshape(nk, 1), u_10, fetch) * spreadf(K.reshape(nk, 1), phi1, u_10, fetch) * K.reshape(nk, 1)**3
    else:
        Bkdir = specf(K.reshape(nk, 1), u_10, fetch, phi1)

    n, alpha = param(K, u_10, fetch)
    lamda = (Bkdir/alpha.reshape(nk, 1))**(n.reshape(nk, 1)+1)/(2*K.reshape(nk, 1)) # distribution of breaking front lengths
    lamda_k = np.trapz(lamda, phi1, axis=1)
    lamda = np.trapz(lamda, K, axis=0)
    lamda_k = np.trapz(lamda_k, K)
    q = const.cq * lamda_k
    nazi = eq_azi.shape[0]
    Awb = np.trapz(np.cos(phi1 - eq_azi.reshape(nazi, 1)) * lamda, phi1, axis=1) / lamda_k
    WB = wb0 * (1 + Mwb * const.theta_wb * Awb)
    return WB, q

def Wb_bi(kr, theta_i, theta_s, theta_eq, eq_azi, u_10, fetch, spec_name, polarization):
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
    if polarization == 'VV':
        poli = 90
    else:
        poli = 0
    bist_ang_az = 2 * eq_azi
    # polarization for the mono-static equivalent
    (rot_ang_1, rot_ang_2, rot_ang_tot, Ps1, Ps2, Ps_tot) = elfouhaily(poli, np.degrees(eq_azi), np.degrees(theta_i), np.degrees(eq_azi), np.degrees(theta_i))
    Ps1_eq_norm = np.linalg.norm(Ps1, axis=-1)
    (rot_ang_1, rot_ang_2, rot_ang_tot, Ps1, Ps2, Ps_tot) = elfouhaily(poli, 0, np.degrees(theta_i), np.degrees(bist_ang_az), np.degrees(theta_s))
    Ps1_bi_norm = np.linalg.norm(Ps1, axis=-1)
    # transfer function
    M = Ps1_bi_norm ** 2 / Ps1_eq_norm ** 2
    #  rotation
    rot = rot_ang_1 - rot_ang_tot
    rot = np.radians(rot)
    WB, q = eq_wb(kr, theta_eq, eq_azi, u_10, fetch, spec_name)
    return M * WB * np.cos(rot) ** 2, q
