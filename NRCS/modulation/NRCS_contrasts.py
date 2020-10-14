import numpy as np
from NRCS import spec
from NRCS import spread
from NRCS.modulation import Spectrum
from NRCS.modulation.Spectrum import Trans_single
from NRCS.spec.kudryavtsev05 import spec_peak, param
import NRCS.constants as const
from NRCS.model import Bragg_scattering
from NRCS.model import Specular_reflection
from NRCS.model import Wave_breaking

def NRCS_before(k, kr, theta, azimuth, u_10, fetch, spec_name, polarization):
    nphi = theta.shape[0]
    nk = k.shape[2]
    nazi = azimuth.shape[0]
    BR = np.zeros([k.shape[0], k.shape[1], nazi, nphi])
    SP = np.zeros([k.shape[0], k.shape[1], nazi, nphi])
    WB = np.zeros([k.shape[0], k.shape[1], nazi, nphi])
    q = np.zeros([k.shape[0], k.shape[1]])
    for ii in np.arange(k.shape[0]):
        for jj in np.arange(k.shape[1]):
            BR[ii, jj, :, :] = Bragg_scattering(k[ii, jj, :].reshape(nk, 1), kr, theta, azimuth, u_10[ii, jj], fetch, spec_name, polarization)
            SP[ii, jj, :, :] = Specular_reflection(kr, theta, azimuth, u_10[ii, jj], fetch, spec_name)
            WB[ii, jj, :, :], q[ii, jj] = Wave_breaking(kr, theta, azimuth, u_10[ii, jj], fetch, spec_name, polarization)

    NRCS = (BR + SP) * (1 - q) + WB * q
    return NRCS

def Bragg_scattering_new(k, K, kr, theta, azimuth, u_10, fetch, spec_name, tsc, polarization):

    nphi = theta.shape[0]
    nk = k.shape[2]
    nazi = azimuth.shape[0]
    kd = const.d*kr
    specf = spec.models[spec_name]
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)

    # B_old, B_new = Spectrum(k, K, u_10, fetch, azimuth, spec_name, tsc)

    # # Sea surface slope in the direction of incidence angle
    phi_inc = np.linspace(-np.pi, np.pi, nazi*2)# in radians wave direction relative to the incidence plane
    nn = 89 * 2 * np.pi / 180
    ni = (np.arange(nk) * nn / nk).reshape(1, nk) - nn / 2
    B_phi_old, B_phi_new = Spectrum(k, K, u_10, fetch, phi_inc, spec_name, tsc)
    inc = np.where(azimuth >= 0)[0]
    incc = np.linspace(1, inc[0], inc[0])
    inc = np.hstack((inc, incc))
    BR = np.zeros([k.shape[0], k.shape[1], nazi, nphi])
    for ii in np.arange(k.shape[0]):
        for jj in np.arange(k.shape[1]):
            Skdir_ni = B_phi_new[ii, jj, :, :] / k[ii, jj, :].reshape(nk, 1) ** 4
            ni = np.trapz(k[ii, jj, :].reshape(nk, 1)**3*np.cos(phi_inc)**2*Skdir_ni, phi_inc, axis=1)
            ni2 = np.trapz(ni[k >= kd], k[k >= kd])
            ni = np.tan(ni).reshape(nk, 1)
            P = np.exp(-0.5 * (ni - np.mean(ni)) ** 2 / ni2) / np.sqrt(2 * np.pi * ni2)
            angle_index = np.logical_and(-3 * 180 * np.arctan(np.sqrt(ni2)) / np.pi < np.arctan(ni) * 180 / np.pi, np.arctan(ni) * 180 / np.pi < 3 * 180 * np.arctan(np.sqrt(ni2)) / np.pi)
            P = P[angle_index]
            ni = ni[angle_index]
            nnk = ni.shape[0]
            ni = ni.reshape(nnk, 1)
            # local incidence angle
            theta_l = np.abs(theta - np.arctan(ni).reshape(nnk, 1))
            kbr = 2*kr*np.sin(theta_l)
            # geometric scattering coefficients [Plant 1997] equation 5,6
            eps_sin = np.sqrt(const.epsilon_sw-np.sin(theta_l)**2)
            if polarization == 'VV':
                G = np.cos(theta_l)**2*(const.epsilon_sw-1)*(const.epsilon_sw*(1+np.sin(theta_l)**2) - np.sin(theta_l)**2) / (const.epsilon_sw*np.cos(theta_l)+eps_sin)**2
                G = np.abs(G)**2
            elif polarization == 'HH':
                G = np.cos(theta_l)**2*(const.epsilon_sw-1)/(np.cos(theta_l)+eps_sin)**2
                G = np.abs(G)**2

            # 3-D Sk computed from kudryavtsev05
            if spec_name == 'kudryavtsev05':
                Skk = np.zeros([nnk, nphi, nazi])
                T = np.zeros([nnk, nphi])
                for nn in np.arange(nnk):
                    T[nn, :] = Trans_single(kbr[nn, :].reshape(nphi, 1), K[ii, jj], u_10[ii, jj], fetch, azimuth, spec_name, divergence[ii, jj])
                    Skk[nn, :, :] = specf(kbr[nn, :].reshape(nphi, 1), u_10[ii, jj], fetch, azimuth) * (1 + np.abs(T[nn, :]))/ kbr[nn, :].reshape(nphi, 1) ** 4
                Skk_pi = Skk[:, :, inc.astype(int)]
            Br = np.zeros([nazi, nphi])
            for num in np.arange(nazi):
                if spec_name == 'elfouhaily':
                    Sk = specf(kbr, u_10[ii, jj], fetch)
                    spreadf = spread.models[spec_name]
                    Skb = Sk * spreadf(kbr, azimuth[num], u_10, fetch) * (1 + np.abs(T))/ kbr  # equation 45
                    Skb_pi = Sk * spreadf(kbr, azimuth[num] + np.pi, u_10, fetch) * (1 + np.abs(T)) / kbr  # equation 45
                else:
                    Skb = Skk[:, :, num]  # equation 45
                    Skb_pi = Skk_pi[:, :, num]
                Skb_r = (Skb+Skb_pi) / 2 # Kudryavtsev 2003a equation 2
                # pure Bragg scattering NRCS
                br0 = 16*np.pi*kr**4*G*Skb_r
                # Bragg scattering composite model
                Brr = br0 * P.reshape(nnk, 1)

                # integral over kbr >= kd
                intebr = []
                for i in np.arange(nphi):
                    a = np.tan(theta[i]-const.d / 2)
                    b = np.tan(theta[i]+const.d / 2)
                    inte = Brr[:, i].reshape(nnk, 1)
                    vv = np.trapz(inte[ni <= a], ni[ni <= a])+np.trapz(inte[ni >= b], ni[ni >= b])
                    intebr.append(vv)
                intebr = np.asarray(intebr)
                Br[num, :] = intebr
            BR[ii, jj, :, :] = Br
    return BR

def Specular_reflection_new(K, kr, theta, azimuth, u_10, fetch, spec_name, tsc):

    nphi = theta.shape[0]
    nazi = azimuth.shape[0]
    # wave number
    kbr = 2*kr*np.sin(theta)
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)

    # Spectral model
    # Omni directional spectrum model name
    specf = spec.models[spec_name]

    # phi = azimuth
    phi = (np.arange(nphi) * 2 * np.pi / nphi).reshape(1, nphi) - np.pi   # in radians wave direction relative to the wind
    theta_ll = 0 # local incidence angle
    #  from Introduction to microwave remote sensing
    cos_theta_ll = np.cos(theta_ll)
    sqrt_e_sin = np.sqrt(const.epsilon_sw + cos_theta_ll**2. - 1.)
    RR_vv = ((const.epsilon_sw*cos_theta_ll - sqrt_e_sin)/(const.epsilon_sw*cos_theta_ll + sqrt_e_sin)) # equation [5.54]
    RR = RR_vv
    SP = np.zeros([u_10.shape[0], u_10.shape[1], nazi, nphi])
    for ii in np.arange(u_10.shape[0]):
        for jj in np.arange(u_10.shape[1]):
            T = Trans_single(kbr.reshape(nphi, 1), K[ii, jj], u_10[ii, jj], fetch, azimuth, spec_name, divergence[ii, jj])
            if spec_name == 'elfouhaily':
                spreadf = spread.models[spec_name]
                SSkdir = specf(kbr.reshape((nphi, 1)), u_10[ii ,jj], fetch) * (1 + np.abs(T)) * spreadf(kbr.reshape((nphi, 1)), phi, u_10[ii ,jj], fetch)/kbr.reshape(nphi, 1) # equation 45
            else:
                SSkdir = specf(kbr.reshape((nphi, 1)), u_10[ii ,jj], fetch, phi) * (1 + np.abs(T)) / kbr.reshape(nphi, 1) ** 4  # equation 45
            # # mean square slope in upwind direction
            Sup = np.trapz(kbr.reshape(nphi, 1)**3*np.cos(phi)**2*SSkdir, phi, axis=1)
            # integration over k<kd
            Sup = np.trapz(Sup[kbr < const.kd], kbr[kbr < const.kd])
            # mean square slope in crosswind direction
            # integration over phi
            Scr = np.trapz(kbr.reshape(nphi, 1)**3*np.sin(phi)**2*SSkdir, phi, axis=1)
           # integration over k<kd
            Scr = np.trapz(Scr[kbr < const.kd], kbr[kbr < const.kd])

            # Kudryavtsev 2005 equation (10) comment
            # Mean squre slope satisfying conditions of the specular reflections
            Ssp = Sup * Scr / (Sup * np.sin(azimuth.reshape(nazi, 1)) ** 2 + Scr * np.cos(azimuth.reshape(nazi, 1)) ** 2)

            SP[ii, jj, :, :] = np.pi*np.abs(RR)**2*np.exp(-np.tan(theta.reshape((1, nphi)))**2/(2*Ssp))/(2*np.pi*np.sqrt(Sup)*np.sqrt(Scr)*np.cos(theta.reshape(1, nphi))**4) # equation [10] in 2005
    return SP

def Wave_breaking_new(K, kr, theta, azimuth, u_10, fetch, spec_name, tsc):
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
    divergence = np.gradient(tsc[:, :, 0], 1e3, axis=1) + np.gradient(tsc[:, :, 1], 1e3, axis=0)

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
    phi1 = (np.arange(nphi) * np.pi / nphi).reshape(1, nphi)-np.pi / 2 # in radians azimuth of breaking surface area: -pi/2,pi/2
    nk = 1024
    nazi = azimuth.shape[0]
    q = np.zeros([u_10.shape[0], u_10.shape[1]])
    WB = np.zeros([u_10.shape[0], u_10.shape[1], nazi, nphi])

    for ii in np.arange(u_10.shape[0]):
        for jj in np.arange(u_10.shape[1]):
            KK = np.linspace(10 * spec_peak(u_10[ii, jj], fetch), knb, nk)
            T = Trans_single(KK.reshape(nk, 1), K[ii, jj], u_10[ii, jj], fetch, azimuth, spec_name,
                             divergence[ii, jj])
            if spec_name == 'elfouhaily':
                spreadf = spread.models[spec_name]
                Bkdir = specf(KK.reshape(nk, 1), u_10[ii, jj], fetch) * (1 + np.abs(T)) * spreadf(KK.reshape(nk, 1), phi1, u_10[ii, jj], fetch) * KK.reshape(nk, 1)**3
            else:
                Bkdir = specf(KK.reshape(nk, 1), u_10[ii, jj], fetch, phi1) * (1 + np.abs(T))
            n, alpha = param(KK, u_10[ii, jj])
            lamda = (Bkdir/alpha.reshape(nk, 1))**(n.reshape(nk, 1)+1)/(2*KK.reshape(nk, 1)) # distribution of breaking front lengths
            lamda_k = np.trapz(lamda, phi1, axis=1)
            lamda = np.trapz(lamda, KK, axis=0)
            lamda_k = np.trapz(lamda_k, KK)
            q[ii, jj] = const.cq * lamda_k
            Awb = np.trapz(np.cos(phi1 - azimuth.reshape(nazi, 1)) * lamda, phi1, axis=1) / lamda_k
            Awb = Awb.reshape(nazi, 1)
            WB[ii, jj, :, :] = wb0.reshape(1, nphi)*(1+Mwb.reshape(1, nphi)*const.theta_wb*Awb)
    return WB, q

def NRCS_after(k, K, kr, theta, azimuth, u_10, fetch, spec_name, tsc, polarization):
    BR = Bragg_scattering_new(k, K, kr, theta, azimuth, u_10, fetch, spec_name, tsc, polarization)
    SP = Specular_reflection_new(K, kr, theta, azimuth, u_10, fetch, spec_name, tsc)
    WB, q = Wave_breaking_new(K, kr, theta, azimuth, u_10, fetch, spec_name, tsc)
    NRCS_new = (BR + SP) * (1 - q) + WB * q
    return NRCS_new

def NRCS_contrasts(k, K, kr, theta, azimuth, u_10, fetch, spec_name, tsc, polarization):
    old = NRCS_before(k, kr, theta, azimuth, u_10, fetch, spec_name, polarization)
    new = NRCS_after(k, K, kr, theta, azimuth, u_10, fetch, spec_name, tsc, polarization)
    return (new - old) / old