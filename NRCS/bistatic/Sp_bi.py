from NRCS import constants as const
from NRCS import spec
from NRCS import spread
import numpy as np
from stereoid.oceans.bistatic_pol import elfouhaily


def eq_sp(kr, theta_eq, eq_azi, u_10, fetch, spec_name):
    """
    :param kr:
    :param theta:
    :param azimuth:
    :param u_10:
    :param fetch:
    :param phi_wind:
    :param spec_name:
    :return:
    """

    nphi = theta_eq.shape[0]
    nazi = eq_azi.shape[0]

    # Spectral model
    # Omni directional spectrum model name
    specf = spec.models[spec_name]

    # phi = azimuth
    phi = (np.arange(nphi) * 2 * np.pi / nphi).reshape(1, nphi) - np.pi   # in radians wave direction relative to the

    theta_ll = 0 # local incidence angle
    #  from Introduction to microwave remote sensing
    cos_theta_ll = np.cos(theta_ll)
    sqrt_e_sin = np.sqrt(const.epsilon_sw + cos_theta_ll**2. - 1.)
    RR_vv = ((const.epsilon_sw*cos_theta_ll - sqrt_e_sin)/(const.epsilon_sw*cos_theta_ll + sqrt_e_sin)) # equation [5.54]
    RR = RR_vv

    SSup = np.zeros([nazi])
    SScr = np.zeros([nazi])
    for num in np.arange(nazi):
        kkbr = 2 * kr * np.sin(theta_eq) * np.cos(eq_azi[num])
        if spec_name == 'elfouhaily':
            spreadf = spread.models[spec_name]
            SSkdir = specf(kkbr.reshape((nphi, 1)), u_10, fetch) * spreadf(kkbr.reshape((nphi, 1)), phi, u_10, fetch)/kkbr.reshape(nphi, 1) # equation 45
        else:
            SSkdir = specf(kkbr.reshape((nphi, 1)), u_10, fetch, phi) / kkbr.reshape(nphi, 1) ** 4  # equation 45
    # # mean square slope in upwind direction
        Sup = np.trapz(kkbr.reshape(nphi, 1)**3*np.cos(phi)**2*SSkdir, phi, axis=1)
    # integration over k<kd
        Sup = np.trapz(Sup[kkbr < const.kd], kkbr[kkbr < const.kd])
    # mean square slope in crosswind direction
    # integration over phi
        Scr = np.trapz(kkbr.reshape(nphi, 1)**3*np.sin(phi)**2*SSkdir, phi, axis=1)
    # integration over k<kd
        Scr = np.trapz(Scr[kkbr < const.kd], kkbr[kkbr < const.kd])
    SSup[num] = Sup
    SScr[num] = Scr
    SSup = SSup.reshape(nazi, 1)
    SScr = SScr.reshape(nazi,1)
    # Kudryavtsev 2005 equation (10) comment
    # Mean squre slope satisfying conditions of the specular reflections
    Ssp = SSup * SScr / (SSup * np.sin(eq_azi.reshape(nazi, 1)) ** 2 + SScr * np.cos(eq_azi.reshape(nazi, 1)) ** 2)

    SP = np.pi*np.abs(RR)**2*np.exp(-np.tan(theta_eq.reshape((1, nphi)))**2/(2*Ssp))/(2*np.pi*np.sqrt(Sup)*np.sqrt(Scr)*np.cos(theta_eq.reshape(1, nphi))**4) # equation [10] in 2005

    return SP

def Sp_bi(kr, theta_i, theta_s, theta_eq, eq_azi, u_10, fetch, spec_name, polarization):
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
    SP = eq_sp(kr, theta_eq, eq_azi, u_10, fetch, spec_name)
    return M * SP * np.cos(rot) ** 2
