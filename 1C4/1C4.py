# ********* “ikke kodemal” ********
# ** Jeg tar emnet på nytt(leverte i ifjor  med kandidate nr 15780 ) **
import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
import matplotlib.pyplot as plt

seed = utils.get_seed("92820")
My_solar_system = SolarSystem(seed)
G = constants.G
solar_mass = constants.m_sun
AU = constants.AU

Masses = [5.16,1.10,0.88,0.91,5.21] # solar mass
files = ["star0_5.16.txt", "star1_1.10.txt", "star2_0.88.txt", "star3_0.91.txt", "star4_5.21.txt"]
lambda_list = []
time_list = []
flux_list = []

for i in range(len(files)):
    file = np.loadtxt(files[i])
    time_of_observation = file[:, 0]
    wavelength = file[:, 1]
    flux = file[:, 2]
    time_list.append(time_of_observation)
    lambda_list.append(wavelength)
    flux_list.append(flux)

time_arrays = np.array(time_list, dtype=object)
lambda_array = np.array(lambda_list, dtype=object)
flux_array = np.array(flux_list, dtype=object)

print(lambda_array.shape)

lambda_0 = 656.3 # in nm
C = constants.c

def velocity():
    v_arrays = C * (lambda_array - lambda_0) / lambda_0
    v_real = np.array(v_arrays)
    print(v_arrays.shape)
    for i in range(len(v_arrays)):
        v_mean = np.mean(v_arrays[i])
        v_real[i] = v_arrays[i] - v_mean
    return v_real

v = velocity()

def theoretical_radial_v(t, t_0, P, v_r):
    v_model = v_r * np.cos((2 * np.pi / P) * (t - t_0))
    return v_model

for i in range(len(Masses)): # plot the velocity  curve for each planet
    plt.plot(time_arrays[i], v[i], label="velocity")
    plt.title(f"Velocity of Star {i}")
    plt.xticks(np.arange(min(time_arrays[i]), max(time_arrays[i]), 1000))
    plt.xlabel("Time[Days]")
    plt.ylabel("Velocity[m/s]")
    plt.grid()
    plt.legend()
    plt.show()

for i in range(len(Masses)):# plottting the flux
    plt.plot(time_arrays[i], flux_array[i], label="flux")
    plt.title(f"Flux of Star {i}")
    plt.xticks(np.arange(min(time_arrays[i]), max(time_arrays[i]), 1000))
    plt.xlabel("Time[Days]")
    plt.ylabel("Flux")
    plt.legend()
    plt.grid()
    plt.show()


def least_square(star, t0_min, t0_max, P_min, P_max, vr_min, vr_max, N=20):
    time_arrays = np.array(time_list, dtype=object)
    v = velocity()
    Delta_t, Delta_P, Delta_vr = [], [], []

    for i, t0min, t0max, Pmin, Pmax, vrmin, vrmax in zip(star, t0_min, t0_max, P_min, P_max, vr_min, vr_max):
        t0_array = np.linspace(t0min, t0max, N) # Create arrays for the t0 with N steps
        P = np.linspace(Pmin, Pmax, N)# Create arrays for the period  with N steps
        vr_array = np.linspace(vrmin, vrmax, N)# Create arrays for the vr with N steps

        t_3d, P_3d, vr_3d = np.meshgrid(t0_array, P, vr_array, indexing='ij')# 3D grid of the combinations of P,vr,t
        t = time_arrays[i]
        # flat the 3d frpm meshgrid to 1d
        vr_model = np.array([theoretical_radial_v(t, t_0, P, v_r) for t_0, P, v_r in zip(t_3d.flatten(), P_3d.flatten(), vr_3d.flatten())])
        vr_model = vr_model.reshape(t_3d.shape + (-1,))
        delta_v = v[i] - vr_model# the differnce between the velocity from the  model and from tha data
        Delta = np.sum(delta_v ** 2, axis=-1) # squaring the differnce
        min_index  = np.argmin(Delta) # finding the index correspond to the minimum value of Delta

        min_idx = np.unravel_index(min_index, Delta.shape )# re writing the minimum index as (t0,P,vr) with the mosst possible values that minimize delta
        Delta_t.append(t0_array[min_idx[0]])# min_idx has the form : (t0,P,vr). t0 is index 0
        Delta_P.append(P[min_idx[1]]) # P is the index 1
        Delta_vr.append(vr_array[min_idx[2]])#P is the index 1

    return Delta_t, Delta_P, Delta_vr

star = [1, 2 , 3]
P_min = [6000, 4500, 6000]
P_max = [7000, 5500, 6500]
vr_min = [15, 5, 25]
vr_max = [25, 15, 35]
t0_min = [500, 2000, 4500]
t0_max = [1500, 3000, 5500]

delta_t0, delta_p, delta_vr = least_square(star, t0_min, t0_max, P_min, P_max, vr_min, vr_max)

for i, t_0, P, v_r in zip(star, delta_t0, delta_p, delta_vr):
    plt.plot(time_arrays[i], v[i], label="velocity")
    plt.title(f" The fit model for star{i}")
    t = time_arrays[i]
    plt.plot(time_arrays[i], theoretical_radial_v(t, t_0, P, v_r), label="Model Radial velocity")
    plt.xticks(np.arange(min(time_arrays[i]), max(time_arrays[i]), 1000))
    plt.xlabel("Time[Days]")
    plt.ylabel("velocity")
    plt.grid()
    plt.legend()
    plt.show()

print(f"delta_t0  : {delta_t0}")
print(f"delta_p = {delta_p}")
print(f"delta_vr = {delta_vr}")

def m_p_sini(M_star, v_star, period):
    return (M_star**(2/3) * v_star * (period)**(1/3)) / ((2 * np.pi * G)**(1/3))

G = constants.G

Masses = [1.10,0.88,0.91] # solar mass


planet1_mass_expected  = m_p_sini(M_star =1.10* solar_mass , v_star = 20, period = 6500* 24 * 3600)
print(f"planet 1 mass expected {planet1_mass_expected}")

planet1_mass_from_model= m_p_sini(M_star = 1.10 * solar_mass, v_star=20.263157894736842, period= 6421.0526315789475 * 24 * 3600)
print(f"planet 1 mass model  = {planet1_mass_from_model}")
ratio2 = planet1_mass_from_model / planet1_mass_expected
print(f"Ratio between expected and calculated mass for planet 1 = {ratio2}")

planet2_mass_expected = m_p_sini(0.88 * solar_mass, 10, 5000 * 24 * 3600)
planet2_mass_from_model = m_p_sini(0.88* solar_mass, 12.368421052631579, 4500.0 * 24 * 3600)
print(f"planet mass expected 2   = {planet2_mass_expected }")
print(f"planet 2  mass model = {planet2_mass_from_model}")
ratio2 = planet2_mass_from_model / planet2_mass_expected
print(f"Ratio between expected and calculated mass for planet 2 = {ratio2}")



planet3_mass_expected = m_p_sini(0.91 * solar_mass, 30, 6250 * 24 * 3600)
planet3_mass_from_model = m_p_sini(0.91 * solar_mass, 31.842105263157894, 6078.9473684210525 * 24 * 3600)
print(f"planet 3 expected  mass = {planet3_mass_expected}")
print(f"planet4mass model = {planet3_mass_from_model}")
ratio3 = planet3_mass_from_model / planet3_mass_expected
print(f"Ratio 3= {ratio3}")

eclipse_days = [(3500,4000),(3500,4000),(6250,6750)]
eclipse_days2 = [(3500,4000),(3500,4000),(250,750)]# zoom in again in the flux plot of star 3 in a diiferent time zone

for s, (start, end) in zip(star, eclipse_days2):
    plt.plot(time_arrays[s][start:end], flux_array[s][start:end], label="Flux in the days when the planet enters the star disc")
    plt.title(f"For star {s}")
    plt.xlabel("Time [Days]")
    plt.ylabel("Flux ")
    plt.legend()
    plt.grid()
    plt.show()

"""
(5,)
(5,)
(5,)
20.263157894736842
605.2631578947369
6421.0526315789475
12.368421052631579
2368.4210526315787
4500.0
31.842105263157894
5078.9473684210525
6078.9473684210525  
delta_t0  : [605.2631578947369, 2368.4210526315787, 5078.9473684210525]
delta_p = [6421.0526315789475, 4500.0, 6078.9473684210525]
delta_vr = [20.263157894736842, 12.368421052631579, 31.842105263157894]
planet 1 mass expected 3.714660309183977e+27
planet 1 mass model  = 3.748238317034292e+27
Ratio between expected and calculated mass for planet 1 = 1.009039321244879
planet mass expected 2   = 1.466565002576152e+27
planet 2  mass model = 1.7513102175159456e+27
Ratio between expected and calculated mass for planet 2 = 1.194157923064858
planet 3 expected  mass = 4.8465391224452016e+27
planet4mass model = 5.096769937249291e+27
Ratio 3= 1.0516308253131033

Process finished with exit code 0



"""
