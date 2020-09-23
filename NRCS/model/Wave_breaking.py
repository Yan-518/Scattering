import numpy as np
from NRCS import constants as const
from NRCS import spec
from NRCS import spread
from NRCS.spec.kudryavtsev05 import param
from NRCS.spec.kudryavtsev05 import spec_peak

def CP_breaking(theta):
    nphi = theta.shape[0]
    theta = theta.reshape(nphi, 1)
    eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta)**2)
    Gvv = np.cos(theta)**2*(const.epsilon_sw-1)*(const.epsilon_sw*(1+np.sin(theta)**2) - np.sin(theta)**2) / (const.epsilon_sw*np.cos(theta)+eps_sin)**2
    Ghh = np.cos(theta)**2*(const.epsilon_sw-1)/(np.cos(theta)+eps_sin)**2
    G = (np.abs(Gvv-Ghh)**2).reshape(nphi, 1)
    Bwb = 1e-2
    WBcp = np.pi * G * const.Swb * Bwb / (2*np.tan(theta)**4*np.sin(theta)**2)
    return WBcp

def Wave_breaking(kr, theta, azimuth, u_10, fetch, spec_name = 'elfouhaily'):
    """
    :param kp:
    :param kr:
    :param theta:
    :param azi:
    :param u_10:
    :param fetch:
    :return:
    """

    nphi = theta.shape[0]

    # Spectral model
    # Omni directional spectrum model name
    specf = spec.models[spec_name]

    # NRCS of plumes
    wb0 = np.exp(-np.tan(theta)**2/const.Swb)/(np.cos(theta)**4*const.Swb)+const.yitawb/const.Swb # Kudryavstev 2003a equation (60)
    knb = min(const.br*kr, const.kwb)

    # tilting transfer function
    dtheta = theta[1]-theta[0]
    Mwb = np.gradient(wb0, dtheta)/wb0

    # distribution function
    # in radians azimuth of breaking surface area: -pi/2,pi/2
    phi1 = np.linspace(-np.pi, np.pi, nphi)
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
    nazi = azimuth.shape[0]
    
    if polarization == 'VH':
        WB = np.ones([nazi, nphi])
        WB = CP_breaking(theta)[:, 0] * WB
    else:
        Awb = np.trapz(np.cos(phi1 - azimuth.reshape(nazi, 1)) * lamda, phi1, axis=1) / lamda_k
        Awb = Awb.reshape(nazi, 1)
        WB = wb0.reshape(1, nphi)*(1+Mwb.reshape(1, nphi)*const.theta_wb*Awb)
    return WB, q
