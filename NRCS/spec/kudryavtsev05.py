import numpy as np
from NRCS import constants as const
from scipy.special import erf
from NRCS import spec
from NRCS import spread

def fv(u_10):
    cg = np.sqrt((0.8 + 0.065*u_10) * 1e-3)  # drag coefficients
    U = cg * u_10  # friction velocity
    return U

def rough(u_10):
    return 0.018 * fv(u_10) ** 2 / const.g + 0.1 * const.v_air / fv(u_10)

def beta(k, u_10, phi):
    """
    Equation (17) in Kudryatvtsev 2005

    :param k:
    :param U:
    :param phi:
    :param z_0: Roughness scale, set to 0.0002 following Wikipedia
    :return:
    """

    U = fv(u_10)  # friction velocity
    # Phase velocity
    c = np.sqrt(const.g / k + const.gamma * k/const.rho_water)
    #
    c_b = 1.5 * (const.rho_air / const.rho_water) * (np.log(np.pi / (k * rough(u_10))) / const.kappa - c / U)
    # (17)
    return c_b * (U / c) ** 2 * np.cos(phi) * np.abs(np.cos(phi))

def beta_v(k, u_10, phi):
    """
    # After (16)
    :param k:
    :param u_10:
    :param phi:
    :param z_0:
    :param v: viscosity coefficient of sea water [m^2/s]
    :return:
    """
    omega = np.sqrt(const.g * k + const.gamma * k ** 3/const.rho_water)

    b_v = beta(k, u_10, phi) - 4 * const.v * k ** 2 / omega  # 2003a

    return np.where(b_v > 0, b_v, 0)

def func_U(k):
    return 1 / 2 * (erf(2 * k) + 1)

def filt_f(k, kk):
    kl = 1.5
    kh = const.ky / const.kwb
    PHI = func_U(k / const.ky - kl) - func_U(k / const.ky - kh)
    f_inf = np.trapz(-PHI / (k / const.ky) ** 2, k / const.ky)
    return np.trapz(-PHI[k < kk] / (k[k < kk] / const.ky) ** 2, k[k < kk] / const.ky) / f_inf

def param(k, u_10):
    #   computing tuning parameters
    n = []
    alpha = []
    ng = 5
    alpha_g = 5e-3
    c = np.sqrt(const.g / k + const.gamma * k / const.rho_water)  # phase velocity
    U = fv(u_10)  # friction velocity
    Cb = 1.5 * (const.rho_air / const.rho_water) * (np.log(np.pi / (k * rough(u_10))) / const.kappa - c / U)
    Cb = np.mean(Cb[const.kwb <= k][k[const.kwb <= k] <= const.ky / 2])
    a = 10 ** (np.log10(alpha_g) + np.log10(Cb) / ng)
    alpha_yita = 10 ** (np.log10(a) - np.log10(Cb))
    for kk in k:
        if kk < const.kwb:
            n.append(ng)
            alpha.append(alpha_g)
        elif kk > const.ky / 2:
            n.append(1)
            alpha.append(alpha_yita)
        else:
            n_int = 1 / ((1 - 1 / ng) * filt_f(k[const.kwb <= k][k[const.kwb <= k] <= const.ky / 2], kk) + 1 / ng)
            n.append(n_int)
            alpha.append(10 ** (np.log10(a) - np.log10(Cb) / n_int))
    return np.asarray(n), np.asarray(alpha)

def spec_peak(u_10, fetch):
    X_0 = 22e3  # Dimensionless fetch
    k_0 = const.g / u_10 ** 2
    # Eq. 4 (below)
    X = k_0 * fetch
    # Eq. 37: Inverse wave age
    Omega_c = 0.84 * np.tanh((X / X_0) ** (0.4)) ** (-0.75)
    # Eq. 3 (below)
    kp = k_0 * Omega_c ** 2
    return kp
# def Isw(k, u_10, fetch, phi):
#     omega = np.sqrt(const.g * k + const.gamma * k ** 3/const.rho_water)
#     n, alpha = param(k, u_10)
#     B_ref = B0(k, u_10, fetch, phi)
#     Lam = (B_ref/alpha) ** (n+1) / (2 * k)
#     SW = []
#     for kk in k:
#         km = min(kk/10, const.kwb)
#         SW.append(const.cb*np.trapz((omega*Lam/k)[k < km], k[k < km]))
#     return np.asarray(SW)
# def B0(k, u_10, fetch, phi):
#     """
#     :param k: wave number
#     :param u_10: wind velocity at the height of 10 m above water
#     :param phi: wave direction relative to the wind
#     :return: background spectrum
#     """
#     nk = k.shape[0]
#     n, alpha = param(k, u_10)
#     B_ref = alpha.reshape(nk, 1) * beta_v(k, u_10, phi) ** (1 / n.reshape(nk, 1))
#     return B_ref

def kudryavtsev05(k, u_10, fetch, phi, return_components = 'False'):
    """
    :param k: wave number
    :param u_10: wind velocity at the height of 10 m above water
    :param phi: wave direction relative to the wind
    :return: background spectrum
    """
    nk = k.shape[0]
    n, alpha = param(k, u_10)
    B0 = alpha.reshape(nk, 1) * beta_v(k, u_10, phi) ** (1 / n.reshape(nk, 1))
    # L_pm = np.exp(-5. / 4. * (spec_peak(u_10, fetch) / k) ** 2)
    # Fm = np.exp(-0.25*(k/const.ky-1.)**2)
    # B0 = Fm * B0
    # B0[(k < 10*spec_peak(u_10, fetch))[:, 0], :] = 0

    # Omni directional spectrum model name
    specf = spec.models['elfouhaily']
    # Directional spectrum model name
    spreadf = spread.models['elfouhaily']

    B_l, B_h, Sk = specf(k.reshape(nk, 1), u_10, fetch, return_components='True')
    B_l= B_l.reshape(nk, 1) * spreadf(k.reshape(nk, 1), phi, u_10, fetch) / k.reshape(nk, 1)
    if return_components == 'False':
        return B0 + B_l
    else:
        return B0
