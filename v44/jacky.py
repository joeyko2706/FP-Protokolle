import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.optimize import curve_fit

# Daten einlesen
thetaReflect, reflectivity_corrected = np.genfromtxt("data.txt", unpack=True)
LAMBDA = 1.54e-10  # m

def fresnel_ideal(a_i):
    """Calculates the reflectivity of an ideal, smooth surface of Si."""
    k = 2 * np.pi / LAMBDA
    n_1 = 1
    delta_2 = 7.6e-6
    beta_2 = LAMBDA / (4 * np.pi) * 141 / 100
    n_2 = 1 - delta_2 - 1j * beta_2

    # Berechnung von k_z1 und k_z2
    k_z1 = k * np.sqrt(n_1**2 - np.square(np.cos(np.deg2rad(a_i))))
    k_z2 = k * np.sqrt(n_2**2 - np.square(np.cos(np.deg2rad(a_i))))

    # Reflexionskoeffizient
    r = (k_z1 - k_z2) / (k_z1 + k_z2)

    # Rückgabe der Reflexivität (Betrag von r zum Quadrat)
    return abs(r)**2


lambda_ka = 1.54e-10 # Angström -> Wellenlaenge der Roentgenstrahlung bei der Kalpha Linie
k = 2*np.pi/lambda_ka   # Wellenvektor
n1 = 1
d1 = 0

##### Startparameter für den Fit #####
d_poly = 4.2e-6 # 1. Schicht Polysterol
d_Sili = 1.6e-5 # 2. Schicht Silizium
b_poly = 1.8e-10
b_sili = .25e-5
d = 8.5e-8
sigma_poly = 4e-10
sigma_sili = 3e-10
######################################


theta_min = 0.2     #Startwert Fit
theta_max = 0.8     #Endwert Fit

x = np.linspace(0, 1.5, 10000)
start_params = [d_poly, d_Sili, b_poly, b_sili, d, sigma_poly, sigma_sili]  # Hier die Startparameter eintragen, die später mit *start_params übergeben werden




def parratt(alpha, delta2, delta3, b2, b3, d2, sigma1, sigma2):
    # alpha is the angle of incidence in degrees
    # delta2, delta3 are the real parts of the refractive indices of the two materials
    # b2, b3 are the imaginary parts of the refractive indices of the two materials
    # d2 is the thickness of the second layer
    # sigma1, sigma2 are the roughnesses of the interfaces    
    
    # Convert angle to radians
    alpha_rad = np.deg2rad(alpha)
    
    # Calculate refractive indices
    n2, n3 = 1.0 - delta2 - 1j * b2, 1.0 - delta3 - 1j * b3
    
    # Calculate wave vectors
    kz1 = k * np.sqrt(n1**2 - np.cos(alpha_rad)**2)
    kz2 = k * np.sqrt(n2**2 - np.cos(alpha_rad)**2)
    kz3 = k * np.sqrt(n3**2 - np.cos(alpha_rad)**2)
    
    # Calculate exponential factors for roughness
    exp_factor1 = np.exp(-2 * kz1 * kz2 * sigma1**2)
    exp_factor2 = np.exp(-2 * kz2 * kz3 * sigma2**2)
    
    # Calculate reflectivities
    r12 = ((kz1 - kz2) / (kz1 + kz2)) * exp_factor1
    r23 = ((kz2 - kz3) / (kz2 + kz3)) * exp_factor2
    
    # Calculate exponential factor for layer thickness
    exp_factor3 = np.exp(-2j * kz2 * d2)
    
    # Calculate total reflectivity
    return np.abs((r12 + exp_factor3 * r23) / (1 + r12 * exp_factor3 * r23))**2


# plt.plot(x, parratt(x, *start_params), label="Parratt-Algorithmus", color="b")
# plt.plot(x, fresnel_ideal(x), label="Fresnelreflektivität", color="gray", linestyle="dashed", alpha=.8)
# plt.errorbar(thetaReflect[-len(thetaReflect)+1:], reflectivity_corrected[-len(reflectivity_corrected)+1:], label="Reflektivität mit Korrekturfaktor", color="red")

# plt.xlabel(r"$\theta\,/\, \text{DEG}$")
# plt.ylabel(r"$R$")
# plt.xlim(0,1.5)
# plt.grid()
# plt.legend(loc="best")
# plt.yscale("log")
# plt.savefig("Parratt.pdf")


fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.flatten()

for i, ax in enumerate(axs):
    ax.plot(x, parratt(x, *start_params[i]), label="Parratt-Algorithmus", color="b")
    ax.plot(x, fresnel_ideal(x), label="Fresnelreflektivität", color="gray", linestyle="dashed", alpha=.8)
    ax.errorbar(thetaReflect[-len(thetaReflect)+1:], reflectivity_corrected[-len(reflectivity_corrected)+1:], label="Reflektivität mit Korrekturfaktor", color="red")
    ax.set_xlabel(r"$\theta\,/\, \text{DEG}$")
    ax.set_ylabel(r"$R$")
    ax.set_xlim(0,1.5)
    ax.grid()
    ax.legend(loc="best")
    ax.set_yscale("log")

plt.tight_layout()
plt.savefig("Parratt.pdf")