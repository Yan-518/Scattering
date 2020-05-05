from NRCS import constants as const
from NRCS import spec
from NRCS import spread
import numpy as np

def Bragg_scattering(k, kr, theta, azimuth, u_10, fetch, spec_name='elfouhaily'):
    """
    :param k:
    :param kr:
    :param theta:
    :param azimuth:
    :param u_10:
    :param fetch:
    :param spec_name:
    :return:
    """
    nphi = theta.shape[0]
    nk = k.shape[0]
    kd = const.d*kr

    # Spectral model
    # Omni directional spectrum model name
    specf = spec.models[spec_name]


    # # Sea surface slope in the direction of incidence angle
    phi_inc = (np.arange(nphi)*2*np.pi/nphi).reshape(1, nphi)-np.pi/2 # in radians wave direction relative to the incidence plane
    if spec_name == 'elfouhaily':
        # Directional spectrum model name
        spreadf = spread.models[spec_name]
        Skdir_ni = specf(k.reshape(nk, 1), u_10, fetch) * spreadf(k.reshape(nk, 1), phi_inc, u_10, fetch)/k.reshape(nk, 1) # equation 45
    else:
        Skdir_ni = specf(k.reshape(nk, 1), u_10, fetch, phi_inc) * k.reshape(nk, 1) ** 4
    ni = np.trapz(k.reshape(nk, 1)**3*np.cos(phi_inc)**2*Skdir_ni, phi_inc, axis=1)
    ni2 = np.trapz(ni[k >= kd], k[k >= kd])

    nn = 89 * 2 * np.pi / 180
    ni = (np.arange(nk) * nn / nk).reshape(1, nk) - nn / 2
    ni = ni.reshape(nk, 1)
    ni = np.tan(ni)
    P = np.exp(-0.5 * (ni - np.mean(ni)) ** 2 / ni2) / np.sqrt(2 * np.pi * ni2)

    # local incidence angle
    theta_l = np.abs(theta - np.arctan(ni).reshape(nk, 1))
    kbr = 2*kr*np.sin(theta_l)

    # geometric scattering coefficients [Plant 1997] equation 5,6
    eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta_l)**2)
    Gvv = np.cos(theta_l)**2*(const.epsilon_sw-1)*(const.epsilon_sw*(1+np.sin(theta_l)**2) - np.sin(theta_l)**2) / (const.epsilon_sw*np.cos(theta_l)+eps_sin)**2
    Gvv = np.abs(Gvv)**2
    Ghh = np.cos(theta_l)**2*(const.epsilon_sw-1)/(np.cos(theta_l)+eps_sin)**2
    Ghh = np.abs(Ghh)**2

    if np.shape(azimuth) == ():
        #  wave number folded folded spcetrum of the surface elevations
        if spec_name == 'elfouhaily':
            Sk = specf(kbr, u_10, fetch)
            spreadf = spread.models[spec_name]
            Skb = Sk * spreadf(kbr, azimuth, u_10, fetch) / kbr  # equation 45
            Skb_pi = Sk * spreadf(kbr, azimuth + np.pi, u_10, fetch) / kbr  # equation 45
        else:
            Skb = specf(kbr, u_10, fetch, azimuth) * kbr ** 4  # equation 45
            Skb_pi = specf(kbr, u_10, fetch, azimuth + np.pi) * kbr ** 4  # equation 45
        Skb_r = (Skb + Skb_pi) / 2  # Kudryavtsev 2003a equation 2

        # pure Bragg scattering NRCS
        br0_vv = 16 * np.pi * kr ** 4 * Gvv * Skb_r
        br0_hh = 16 * np.pi * kr ** 4 * Ghh * Skb_r

        # Bragg scattering composite model
        BR_vv = br0_vv * P.reshape(nk, 1)
        BR_hh = br0_hh * P.reshape(nk, 1)

        # integral over kbr >= kd
        VV = []
        HH = []

        for i in np.arange(nphi):
            a = np.tan(theta[i] - const.d / 2)
            b = np.tan(theta[i] + const.d / 2)
            inte_vv = BR_vv[:, i].reshape(nk, 1)
            inte_hh = BR_hh[:, i].reshape(nk, 1)
            vv = np.trapz(inte_vv[ni <= a], ni[ni <= a]) + np.trapz(inte_vv[ni >= b], ni[ni >= b])
            hh = np.trapz(inte_hh[ni <= a], ni[ni <= a]) + np.trapz(inte_hh[ni >= b], ni[ni >= b])
            VV.append(vv)
            HH.append(hh)
        VV = np.asarray(VV)
        HH = np.asarray(HH)
        return VV, HH
    else:
        nazi = azimuth.shape[0]
        Br_VV = np.zeros([nazi, nphi])
        Br_HH = np.zeros([nazi, nphi])
        for num in np.arange(nazi):
            #  wave number folded folded spcetrum of the surface elevations
            if spec_name == 'elfouhaily':
                Sk = specf(kbr, u_10, fetch)
                spreadf = spread.models[spec_name]
                Skb = Sk * spreadf(kbr, azimuth[num], u_10, fetch) / kbr  # equation 45
                Skb_pi = Sk * spreadf(kbr, azimuth[num] + np.pi, u_10, fetch) / kbr  # equation 45
            else:
                Skb = np.zeros([kbr.shape[0], kbr.shape[1]])
                Skb_pi = np.zeros([kbr.shape[0], kbr.shape[1]])
                for h in np.arange(kbr.shape[0]):
                    Skb[h, :] = (specf(kbr[h, :].reshape(kbr.shape[1], 1), u_10, fetch, azimuth[num]) * kbr[h, :].reshape(kbr.shape[1], 1) ** 4)[:, 0]  # equation 45
                    Skb_pi[h, :] = (specf(kbr[h, :].reshape(kbr.shape[1], 1), u_10, fetch, azimuth[num] + np.pi) * kbr[h, :].reshape(kbr.shape[1], 1) ** 4)[:, 0]  # equation 45
            Skb_r = (Skb+Skb_pi) / 2 # Kudryavtsev 2003a equation 2

        # pure Bragg scattering NRCS
            br0_vv = 16*np.pi*kr**4*Gvv*Skb_r
            br0_hh = 16*np.pi*kr**4*Ghh*Skb_r

        # Bragg scattering composite model
            BR_vv = br0_vv*P.reshape(nk, 1)
            BR_hh = br0_hh*P.reshape(nk, 1)

        # integral over kbr >= kd
            VV = []
            HH = []

            for i in np.arange(nphi):
                 a = np.tan(theta[i]-const.d / 2)
                 b = np.tan(theta[i]+const.d / 2)
                 inte_vv = BR_vv[:, i].reshape(nk, 1)
                 inte_hh = BR_hh[:, i].reshape(nk, 1)
                 vv = np.trapz(inte_vv[ni <= a], ni[ni <= a])+np.trapz(inte_vv[ni >= b], ni[ni >= b])
                 hh = np.trapz(inte_hh[ni <= a], ni[ni <= a])+np.trapz(inte_hh[ni >= b], ni[ni >= b])
                 VV.append(vv)
                 HH.append(hh)
            VV = np.asarray(VV)
            HH = np.asarray(HH)
            Br_VV[num, :] = VV
            Br_HH[num, :] = HH
        return Br_VV, Br_HH