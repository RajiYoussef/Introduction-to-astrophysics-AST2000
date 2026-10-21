
# ********* “ikke kodemal” ********
# ** Jeg tar emnet på nytt(leverte i ifjor  med kandidate nr 15780 ) **
import numpy as np
import ast2000tools.utils as utils
import ast2000tools.constants as constants
from ast2000tools.solar_system import SolarSystem
import matplotlib.pyplot as plt
from numpy import linalg as LA

seed = utils.get_seed("92820")
np.random.seed(seed)
system = SolarSystem(seed)

my_home_planet = system.masses[0]
R = system.radii[0]

print(f"The mass of my planet is :{my_home_planet} solar mass with radius :{R}Km")

print("\nPart 1 : calculating the escape velocity:")

"""
The escape velocity is give by the formula = v = sqrt(2Mg/R). M is the planet mass and R is its radius. 
"""

g = constants.G
M = my_home_planet * constants.m_sun  # here i converted the mass from solar mass to Kg.
earth_mass = 5.972 * 10 ** 24  # Kg
ratio = M / earth_mass
print(f"the ration is between my home planet and earth  is  : {ratio:.3f}")
print(f"The mass of my planet in Kg :{M} Kg")

R = R * 1e3  # Also ,here i converted the radius to meters
print(f"The radius  of my home planet in meters is : {R}m")


def escape_velocity(M, R):
    v_es = np.sqrt((2 * g * M) / R)
    return v_es


es_velocity = escape_velocity(M, R)
print(f"The escape velocity from my home planet with mass {my_home_planet} solar mass  is {es_velocity:.4f} m/s = {es_velocity * 1e-3:.3f} Km/s ")

print("\n=============Part 2 :=========================")
L = 10 ** -6  # the length of the box in meters
T = 10000  # gas temperature in Kelvin
N = 10 ** 5  # particles number


def position(N):
    x_y_z = np.random.uniform(-L / 2, L / 2, (N, 3)) # random function locates the particles in random positions with uniform distibution
    return x_y_z
posision_arrays = position(N)


def mean_possion(posision_arrays):
    pos_magnitude = LA.norm(posision_arrays, axis=0) # we know that r = sqrt(x^2 + y^2 + x^2)
    mean = np.sum(pos_magnitude) / N
    return mean
Mean_value = mean_possion(posision_arrays)
print(f"The mean value of the position is = {Mean_value} m")

k = constants.k_B # Boltzmann constant
m = constants.m_H2 # The mass of hydrogen molecules
print(m)

def velocities(N):
    sigma = np.sqrt(k * T / m) # The standard deviation of the normal distribution.
    vx_vy_vz = np.random.normal(0, sigma, (N, 3)) # this attribute generate for us random velocity vectors following the
    # the normal distibution with a range of standard deviation.
    return vx_vy_vz
particles_velocity = velocities(N)


def mean_velocity(particles_velocity):
    magnitude = LA.norm(particles_velocity, axis=1) # v = sqrt(vx^2 + vy^2 +vz^2)
    mean = np.mean(magnitude)
    return mean
The_mean_velocity = mean_velocity(particles_velocity)
print(f"The mean velocity of the particleis : {The_mean_velocity}m/s")

theoretical_mean_velocity = np.sqrt((8 * k * T) / (np.pi * m))
print(f"THeoretical mean velocity is {theoretical_mean_velocity} m/s")
print(f"Ratio between calculated and theoretical velocity  is {The_mean_velocity / theoretical_mean_velocity}")
# Here i just want to plot the the distribution of the velocities
histo = np.linalg.norm(particles_velocity, axis=1) # gives us the magnitude of v disributed as  histograms
plt.hist(histo, bins=50)
plt.title('Speed Distribution')
plt.axvline(theoretical_mean_velocity, label="Theoretical mean velocity", color="red")
plt.axvline(The_mean_velocity, label="The calculated mean velocity", color="black")
plt.xlabel('velocity (m/s)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()


def knetic(N):
    E = 1 / 2 * m * (mean_velocity(particles_velocity) ** 2) * N# E = 1/2mv^2, we considered the mean v
    return E
kinetic_energy = knetic(N)
print(f"The total kinetic energy for {N} particles  is : {kinetic_energy} joul")

def mean_kinetic(N):
    particles_velocity = velocities(N)
    velocity_magnitude = LA.norm(particles_velocity, axis=1)
    E_mean = 1 / 2 * m * np.mean(velocity_magnitude ** 2)
    return E_mean
mean_Ek = mean_kinetic(N)
print(f"The mean kenetic energy is {mean_Ek}")

theoretical_mean_E = (3 / 2) * k * T
print(f"The theoretical mean kinetic energy  is {theoretical_mean_E}")
print(f"Ratio between calculated and theoretical kinetic is {mean_Ek / theoretical_mean_E}")


print("\n=======Part 3 : Making the engine ======")

def initials(N):
    vel = velocities(N)
    pos = position(N)
    return vel, pos


def escap(N, dt, steps, vel, pos):
    escaped_particles = 0 # intial number of particles
    v = 0 # Giving initial value for velocities
    for j in range(steps):
        pos += vel * dt # simple formula s = v*t
        i = np.where(abs(pos) > L / 2) # by assuming that the origin is at the middle of the box. when the position of
        # a particle is > L/2, means this particle is out of the box. abs : do it in both sides .
        vel[i] *= (-1) # here :when a particle collides with the wall, we reverse its velocity.
        v += np.sum(abs(vel[i])) # after reversing the velocities ,we take the sum of them.
        escaped_particles += len(i[0]) # then we sum of all particles that collided with the wall
    return escaped_particles, v


Delta_t = 10 ** -9
dt = 10 ** -12
steps = 1000
A = L ** 2 * 6 # the area of the box
A_ratio = A / ((L / 2) ** 2)# the ratio between the area of the box and the hole
print(f"ratio between hole and the side area {A_ratio} ")

def calculate(N, escaped_particles, v, steps, dt, A):
    vol_box = L ** 3  # m^3
    n = N / vol_box
    A_ratio = A / ((L / 2) ** 2)  # (L /2)**2 is the area of the hole, A_ratio is ratio between hole and the side area.
    fuel_loss = (escaped_particles  * m / A_ratio) # the mass of fuel we loose per hole
    momentum = m * 2 * v  # elastic collision between the particle and the wall yields that the particle velocity is 2v
    change_in_momentum = v * m / A_ratio # the change in momentum per particle per hole
    F = momentum / (Delta_t)
    actual_p = F / A
    theoretical_pressure = n * k * T  # from ideal gas law
    return theoretical_pressure, actual_p, fuel_loss, change_in_momentum


def simulation(N):
    vel, pos = initials(N)
    particles_escaping, v = escap(N, dt, steps, vel, pos)
    return calculate(N, particles_escaping, v, steps, dt, A)


threoretical_P, actual_P, Loss_in_fuel, momentum_change = simulation(N)
print(f"The theoretical pressure is {threoretical_P} while the calculated is {actual_P}")
print(f"The fuel loss is :{Loss_in_fuel}")
print(f" The ratio between actual and theoretical pressure is {threoretical_P / actual_P}")
print(f"The change in momentum : {momentum_change}")

print("\n=======Part 4 : The accleration  ======")



def a_satatlit(es_velocity):
    fuel_loss = simulation(N)[2]
    t = 1200 # seconds
    m_satalite = 1000 # kg
    a_by_box = momentum_change / (m_satalite * Delta_t)
    a_total = es_velocity / t # V = at
    n_boxes = a_total / a_by_box
    total_fuel = fuel_loss * n_boxes * t / Delta_t
    return a_total, n_boxes, total_fuel


accelration, number_of_boxes, fuel = a_satatlit(es_velocity)
print(f"Number of boxes needed to acclerate 1000 kg satalite to {accelration} m/s^2is {number_of_boxes} Box")

print(f"the fuel calculated  to accelrate the satalite is {fuel:.3f} Kg")

m_satalite = 1000

m_fuel = np.sqrt(3) * es_velocity * m_satalite / theoretical_mean_velocity

print(f"The mass of fuel needed : {m_fuel:.3f} kg")
print(f" the ration between calculated fuel and theoretical {(fuel / m_fuel) * 100}%")

"""

The mass of my planet is :9.565492243971476e-06 solar mass with radius :9386.516965328467Km

Part 1 : calculating the escape velocity:
the ration is between my home planet and earth  is  : 3.185
The mass of my planet in Kg :1.9020746162744973e+25 Kg
The radius  of my home planet in meters is : 9386516.965328466m
The escape velocity from my home planet with mass 9.565492243971476e-06 solar mass  is 16446.7238 m/s = 16.447 Km/s 

=============Part 2 :=========================
The mean value of the position is = 2.7368352603350683e-09 m
3.347447494734414e-27
The mean velocity of the particleis : 10229.403569577591m/s
THeoretical mean velocity is 10248.370356913738 m/s
Ratio between calculated and theoretical velocity  is 0.998149287479316
The total kinetic energy for 100000 particles  is : 1.751396201615149e-14 joul
The mean kenetic energy is 2.0630656043545227e-19
The theoretical mean kinetic energy  is 2.0709735000000001e-19
Ratio between calculated and theoretical kinetic is 0.996181556333059

=======Part 3 : Making the engine ======
ratio between hole and the side area 24.0 
The theoretical pressure is 13806.490000000002 while the calculated is 13639.358173330344
The fuel loss is :2.1259192712725632e-22
 The ratio between actual and theoretical pressure is 1.0122536430633853
The change in momentum : 1.704919771666293e-18

=======Part 4 : The accleration  ======
Number of boxes needed to acclerate 1000 kg satalite to 13.705603190245881 m/s^2is 8038855210677.037 Box
the fuel calculated  to accelrate the satalite is 2046.378 Kg
The mass of fuel needed : 2779.619 kg
 the ration between calculated fuel and theoretical 73.6208059800473%

Process finished with exit code 0



"""