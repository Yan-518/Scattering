"""
    **Custom Added Constants**
 """

import numpy as np
from scipy.constants import g

# Sea water rel.dielectric constant
epsilon_sw = np.complex(73, 18)
# Constant in close agreement with [Thompson, 1988] Kudryavtsev [2005]
d = 1 / 4
# C-band radio wave number. Estimate of sea state from Sentinel-1 SAR imagery for maritime situation awareness 1.1, 5.6 cm
kr =2 * np.pi / 5.6e-2
# Divide two intervals (small and large scale waves) [Kudryavstev, 2005]
kd = d * kr
# Mean square slope of enhanced roughness of the wave breaking zone [Kudryavstev, 2003]
Swb = 0.19
# Ratio of the breaker thickness to its length [Kudryavstev, 2003]
yitawb = 0.005
# Wave break wave number [Kudryavtsev, 2005]
kwb = 2 * np.pi / 0.3
# Density of water kg /mˆ3
rho_water = 1e3
# Density of air kg /mˆ3 at 15 degree
rho_air = 1.225
# Von Karman constant
kappa = 0.4
# Surface tension of water at 20 degree
gamma = 0.07275
# Wave number of minimum phase velocity
ky = np.sqrt(g * rho_water / gamma)
# Saturation level constant [Kudryavtsev, 2003]
a = 2.5e-3
# Tunning parameter [Kudryavstev, 2003]
alphag = 5e-3
# The constant of the equilibrium gravity range [Kudryavstev, 2003]
ng = 5
# Mean tilt of enhances scattering areas of breaking waves [Kudryavstev, 2003]
theta_wb = 5e-2
# Constant to compute fraction q [Kudryavstev, 2003]
cq = 10.5
# Constant implies that the length of waves providing non-Bragg scattering are more than 10 ten times longer than the radar wave length [Kudryavtsev, 2003]
br = 0.1
# Averaged tilt of parasitic capillary trains [Kudryavtsev, 2003]
theta_pc = 5e-2
# Coriolis parameter f-plane approximation (doesn't change with altitude)
f = 1e-4
# kinematic viscosity coefficient of sea water at 20 degree [m^2/s]
v = 1.15e-6
# kinematic viscosity coefficient of air at 15 degree [m^2/s]
v_air = 1.47e-5
# empirical constant for wave breaking
cb = 1.2e-2
# thermal expansion coefficient unit K^(-1)
alpha = 207e-6
# constant for computing divergence
yita = 0.2
# Prandtl ratio for the Brunt-Vaisala frequency
nb = 50
