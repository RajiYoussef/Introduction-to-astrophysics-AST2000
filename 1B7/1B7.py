# ********* “ikke kodemal” ********
# ** Jeg tar emnet på nytt(leverte i ifjor  med kandidate nr 15780 ) **


import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
import matplotlib.pyplot as plt
from numpy import linalg as LA


seed = utils.get_seed("92820")
My_solar_system = SolarSystem(seed)
print(f"First : printing information about my solar system:")
eccentricity = My_solar_system.eccentricities
aphelion_angles = My_solar_system.aphelion_angles - np.pi
lis_of_masses = My_solar_system.masses
semi_majors = My_solar_system.semi_major_axes
print(f"\n")
print(f"List of Aphelion angles for my system :{aphelion_angles} rad")
print(f"List of masses for my solar system :{lis_of_masses}in solar mass. ")
print(f"List of Eccentricity for my solar system : {eccentricity}")
print( f"List of semi majors for my solar system :{semi_majors} in AU.")

My_sun_mass = My_solar_system.star_mass
My_star_radius = My_solar_system.star_radius
planets_list = My_solar_system.number_of_planets
print(f"Number of planets in my solar system is : {planets_list} planets .")
V_initials = My_solar_system.initial_velocities
Pos_initial = My_solar_system.initial_positions

velocities_0 = np.array(V_initials)
Posision_0 = np.array(Pos_initial)
print(f"The initial velocities of my planets : {velocities_0}")
print(f"Initial positions pf my planets :{Posision_0}")

g = constants.G
solar_mass = constants.m_sun
AU = constants.AU
Solar_G = constants.G_sol

print(f"My solar system Sun  has a mass with {My_sun_mass * constants.m_sun} Kg , with a radius : {My_star_radius } AU")
print(f"\n")
for i in range(planets_list):
    planet_index = [i]
    type = My_solar_system.types[i]
    semi_major = semi_majors[i]
    planet_mass = lis_of_masses[i]
    print(f"The planet{i} , is a {type} planet ,hasa mass {planet_mass * constants.m_sun}Kg , with a semi major {semi_major:.4f} AU")

for i in range(planets_list):
    print(f" The planet{i},has angle = {aphelion_angles[i]:.4f} with Eccentricity : {eccentricity[i]:.4f}")


def My_system(dt, period, N):
    pos_array = np.zeros((2, planets_list, N)) # Initial position array for x and y of every planet
    vel_array = np.zeros((2, planets_list, N)) # Initial velocity array for x and y of every planet
    pos_array[:, :, 0] = Pos_initial  # initial positions
    vel_array[:, :, 0] = V_initials  # initial velocities

    for i in range(1, N):
        r = np.linalg.norm(pos_array[:, :, i - 1], axis=0)  # Calculate distances for everyplanet
        a = -Solar_G * My_sun_mass * pos_array[:, :, i - 1] / r ** 3  #  accelerations
        vel_array[:, :, i] = vel_array[:, :, i - 1] + a * dt  # Update the velocity for each planet based on acceleration .
        # vel_array[:, :, i] is v_{n+1} , vel_array[:, :, i - 1] is v_{n}

        pos_array[:, :, i] = pos_array[:, :, i - 1] + vel_array[:, :, i] * dt
        #Update positions using Euler-Cromer. in this step pos_array[:, :, i] is :r_{n+1} , pos_array[:, :, i - 1] is r_n
         #and vel_array[:, :, i] is v_{n+1} as the general form of Euler-Cromer : r_{n+1} = r_n + v_{n+1} *dt


    return vel_array, pos_array


for p, semi_major in enumerate(semi_majors):
    T = np.sqrt(4 * np.pi ** 2 * semi_major ** 3 / (Solar_G * My_sun_mass))
    print(f"Period of planet {p} with a semi-major axis of {semi_major:.4f} is {T:.4f}")

dt = 0.01
period = 203
N = int(period / dt)
velocities,pos  = My_system(dt, period ,N)

time = np.linspace(0,20, N)
My_solar_system.generate_orbit_video(time , pos, filename='92820_system_video.xml')

f = np.linspace(0, 2 * np.pi, 1000)
types = ["rock","rock","gas","rock","gas","rock","rock"]

# The Analytical method :
plt.figure(figsize=(10, 6))
for i in range(planets_list):
    R = semi_majors[i] * (1 - eccentricity[i] ** 2) / (1 + eccentricity[i] * np.cos(f))
    theta = f + aphelion_angles[i]
    x, y = R * np.cos(theta), R * np.sin(theta)
    plt.plot(x, y, label=f"Planet {i}: {types[i]}")
    plt.plot(Pos_initial[0, i], Pos_initial[1, i], "o")

plt.plot(0, 0, 'o', color="black", label="star")
plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")
plt.title("The analytical plot of planets orbits of my solar system")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()


# The numerical method :
plt.figure(figsize=(10, 6))

for j in range(planets_list):
    plt.plot(pos[0, j, :], pos[1, j, :], label=f"Planet {j}, {types[j]}")
    plt.plot(Pos_initial[0, j], Pos_initial[1, j], "o")

plt.plot(0, 0, marker='o', color="black", label="Star")
plt.title("Numerical plot of planets orbits of my solar system after time")
plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()


# Comining both methods to check if there is any deviation
plt.figure(figsize=(10, 6))

for i in range(planets_list):
    R = semi_majors[i] * (1 - eccentricity[i] ** 2) / (1 + eccentricity[i] * np.cos(f))
    theta = f + aphelion_angles[i]
    x, y = R * np.cos(theta), R * np.sin(theta)
    plt.plot(x, y, linestyle='--', label=f"Analytical Planet {i}: {types[i]}")

for i in range(planets_list):
    plt.plot(pos[0, i, :], pos[1, i, :], label=f"Numerical Planet {i}, {types[i]}")
    plt.plot(Pos_initial[0, i], Pos_initial[1, i], "o")

plt.plot(0, 0, marker='o', color="black", label="Star")
plt.title("comparing  the  analytical and numerical orbits of my solar system ")
plt.xlabel("X [AU]")
plt.ylabel("Y [AU]")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()




"""

First : printing information about my solar system:


List of Aphelion angles for my system :[-3.14159265 -1.51740896  1.7006222   1.0805196  -0.10351786  1.53750664
 -1.60178598] rad
List of masses for my solar system :[9.56549224e-06 7.24139825e-06 1.68886985e-02 2.05324269e-07
 4.31522099e-04 1.57160735e-06 2.90807405e-07]in solar mass. 
List of Eccentricity for my solar system : [0.00027726 0.00210866 0.05573695 0.05193469 0.03136916 0.06272496
 0.03673278]
List of semi majors for my solar system :[ 6.65626189  8.98216146 55.0812933   4.47044201 26.55997189 20.55324904
 12.323441  ] in AU.
Number of planets in my solar system is : 7 planets .
The initial velocities of my planets : [[ 0.         -4.10408674  1.60481973 -6.02261873 -0.96830113  0.0740206
   0.49157132]
 [ 4.88033528  0.86291072  0.04278734  1.73462467 -2.16540371 -2.76663526
   3.56799219]]
Initial positions pf my planets :[[  6.65810743   1.84712722   1.88299721   1.13023893 -25.0769353
  -20.54836408  12.21376318]
 [  0.           8.80888412 -58.10457922   4.09005955  10.92066788
   -1.84196018  -1.23009763]]
My solar system Sun  has a mass with 7.989710375255283e+30 Kg , with a radius : 2070487.2730332096 AU


The planet0 , is a rock planet ,hasa mass 1.9020746162744973e+25Kg , with a semi major 6.6563 AU
The planet1 , is a rock planet ,hasa mass 1.4399342400656765e+25Kg , with a semi major 8.9822 AU
The planet2 , is a gas planet ,hasa mass 3.358276179845791e+28Kg , with a semi major 55.0813 AU
The planet3 , is a rock planet ,hasa mass 4.082822613462485e+23Kg , with a semi major 4.4704 AU
The planet4 , is a gas planet ,hasa mass 8.580710844586864e+26Kg , with a semi major 26.5600 AU
The planet5 , is a rock planet ,hasa mass 3.125102582914652e+24Kg , with a semi major 20.5532 AU
The planet6 , is a rock planet ,hasa mass 5.7826337505716286e+23Kg , with a semi major 12.3234 AU
 The planet0,has angle = -3.1416 with Eccentricity : 0.0003
 The planet1,has angle = -1.5174 with Eccentricity : 0.0021
 The planet2,has angle = 1.7006 with Eccentricity : 0.0557
 The planet3,has angle = 1.0805 with Eccentricity : 0.0519
 The planet4,has angle = -0.1035 with Eccentricity : 0.0314
 The planet5,has angle = 1.5375 with Eccentricity : 0.0627
 The planet6,has angle = -1.6018 with Eccentricity : 0.0367
Period of planet 0 with a semi-major axis of 6.6563 is 8.5672
Period of planet 1 with a semi-major axis of 8.9822 is 13.4297
Period of planet 2 with a semi-major axis of 55.0813 is 203.9392
Period of planet 3 with a semi-major axis of 4.4704 is 4.7154
Period of planet 4 with a semi-major axis of 26.5600 is 68.2867
Period of planet 5 with a semi-major axis of 20.5532 is 46.4853
Period of planet 6 with a semi-major axis of 12.3234 is 21.5820
Generating orbit video with 100 frames.
Note that planet/moon rotations and moon velocities are adjusted for smooth animation.
XML file 15780_system_video.xml was saved in XMLs/.
It can be viewed in SSView.

Process finished with exit code 0



"""

