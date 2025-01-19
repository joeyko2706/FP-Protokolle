import matplotlib.pyplot as plt         # type: ignore
import numpy as np                      # type: ignore
import uncertainties.unumpy as unp      # type: ignore
from uncertainties import ufloat        # type: ignore
from scipy.optimize import curve_fit    # type: ignore


#######################################################
### Bestimmung der Halbwertsbreite und der maximalen Intensität
#######################################################

def gaussian_fit(alpha, alpha0, sigma, I0, B):
    return I0/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(alpha-alpha0)**2/(2*sigma**2)) + B


theta, hits = np.genfromtxt("./data/Detector1.UXD", unpack=True, skip_header=56)

bounds = [[-0.5, 0, 0, -0.001], [0.5, 0.5, 2e5, 1e4]]

params, covariance = curve_fit(gaussian_fit, theta, hits, bounds=bounds)

errors = np.sqrt(np.diag(covariance))
sigma = ufloat(params[1], errors[1])
FWHM = 2*unp.sqrt(2*np.log(2))*sigma        # Standardformel der FWHM
hline = params[2]/(2*np.sqrt(2*np.pi*params[1]**2)) - params[3]
vline = [params[0] - 1/2*FWHM.nominal_value, params[0] + 1/2*FWHM.nominal_value]

x = np.linspace(-0.5, 0.5, 1000)
plt.plot(x, gaussian_fit(x, *params), color = "b", label="Fit")
plt.plot(theta, hits, "x", color="r", label="Messwerte")
plt.hlines(hline, xmin = -0.5, xmax = 0.5, color = "gray", ls = "dashed", label = r"$\frac{1}{2} \;I_0$")
plt.vlines(vline , 0, 400000, color = "gray", label = f"FWHM: {FWHM.nominal_value:.3f}({FWHM.std_dev:.3f})°")

plt.xlim(-0.5, 0.5)
plt.ylim(-0.9, 400000)
plt.xlabel(r"$\alpha/ \text{DEG} $")
plt.ylabel(r"$I / \text{Hits/s}$")
plt.grid()
plt.legend(loc="best")
plt.savefig("build/DetectorScan.pdf")
plt.clf()

I0 = ufloat(params[2], errors[2])
print("\n")
print("--------------Parameter der Gaußfunktion---------------")
print(f"alpha0  : {params[0]:.4e} +/- {errors[0]:.4e}")
print(f"sigma   : {params[1]:.4e} +/- {errors[1]:.4e}")
print(f"I0      : {I0:.4e}")
print(f"B       : {params[3]:.4e} +/- {errors[3]:.4e}")
print(f"FWHM    : {FWHM}")


#######################################################
### Bestimmung der Strahlbreite (mit erstem Zscan)
####################################################### 25 28

z, hits = np.genfromtxt("./data/Zscan1.UXD", unpack=True, skip_header=56)
upperLimit = 37
lowerLimit = 32
strahlbreite = np.round(z[upperLimit]- z[lowerLimit],2)

plt.errorbar(z, hits, yerr=np.sqrt(hits), fmt="rx", label="Messwerte")
plt.vlines([z[lowerLimit], z[upperLimit]], ymin=0, ymax= 4e5, color="b", label = f"Strahlbreite: {strahlbreite} $mm$")    # counted 25 + 3 = 28

plt.xlim(-1, 1)
plt.ylim(-0.9, 400000)
plt.xlabel(r"$z\,/\text{mm}$")
plt.ylabel(r"$I\,/\,\text{Hits/s}$")
plt.grid()
plt.legend(loc="best")
plt.savefig("build/ZScan.pdf")
plt.clf()

print("-------------------------------------------------------")
print(f"Strahlbreite: {strahlbreite} mm")


#######################################################
### Bestimmung des Geometriewinkels (mit RockingScan)
#######################################################

theta, hits = np.genfromtxt("./data/Rocking1_1.UXD", unpack=True, skip_header=56)


lowerLimit = 16
upperLimit = lowerLimit + 10
# print(f"Schrittweite bei dem Rockingscan: {theta[1]-theta[0]}") # --> 0.04 (alles richtig)

plt.errorbar(theta, hits, yerr=np.sqrt(hits), fmt="rx", label="Messwerte") 
plt.vlines([theta[lowerLimit], theta[upperLimit]], ymin=0, ymax= 175e3, color="b", label = f"Geometriewinkel: {theta[upperLimit]- theta[lowerLimit]:.2}°") # counted 16 36

plt.xlim(-1, 1)
plt.ylim(-0, 17.5e4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0), useMathText=True)
plt.xlabel(r"$\theta\,/\, \text{DEG}$")
plt.ylabel(r"$I\,/\,\text{Hits/s}$")
plt.grid()
plt.legend(loc="upper left")
plt.savefig("build/RockingScan.pdf")
plt.clf()


winkel_theorie = np.arcsin(strahlbreite/20) # rad
winkel_experiment = (theta[upperLimit]- theta[lowerLimit])
print("-----------------Geometriewinkel-----------------------")
print(f"Experiment: {winkel_experiment:.2}°")
print(f"Theorie:    {np.rad2deg(winkel_theorie):.2}°")
print("-------------------------------------------------------")


def geom_factor(thetaReflect, strahlbreite, winkel_theorie):
    factor = np.ones(len(thetaReflect))
    for index, angle in enumerate(thetaReflect):
        if thetaReflect[index] < winkel_theorie:
            print(index)
            factor[index] = 20 * np.sin(np.deg2rad(thetaReflect)) / strahlbreite
    return factor


def factor(thetaReflect, strahlbreite, winkel_theorie):
    factor = np.ones(len(thetaReflect))
    for i in range(len(thetaReflect)):
        if thetaReflect[i] < winkel_theorie:
            factor[i] = 20 * np.sin(np.deg2rad(thetaReflect[i])) / strahlbreite
    return factor



#######################################################
### Bestimmung der Dispersion und Rauigkeit des Siliziumwafers
#######################################################

thetaReflect, hitsReflect = np.genfromtxt("./data/Reflect1.UXD", unpack=True, skip_header=56)
thetaDiffuser, hitsDiffuser = np.genfromtxt("./data/Reflect2.UXD", unpack=True, skip_header=56)

# print(thetaReflect.size, thetaDiffuser.size)


plt.plot(thetaReflect, hitsReflect, label="Refelct", color="r")
plt.plot(thetaDiffuser[:thetaReflect.size], hitsDiffuser[:thetaReflect.size], label="Diffuser", color="b")
plt.plot(thetaReflect, hitsReflect - hitsDiffuser[:thetaReflect.size], linestyle="dashed", label="Differenz", color="darkred")

plt.xlim(0,1.5)
plt.yscale("log")
plt.xlabel(r"$\theta\,/\, \text{DEG}$")
plt.ylabel(r"$I\,/\,\text{Hits/s}$")
plt.grid()
plt.legend(loc="best")
plt.savefig("build/ReflectDiffuserScan.pdf")
plt.clf()

#######################################################
### Schichtdicke des Siliziumwafers
#######################################################




geom_factor = factor(thetaReflect, strahlbreite, np.rad2deg(winkel_theorie))
critical_angle = 0.223 # deg
reflectivity = (hitsReflect - hitsDiffuser[thetaReflect.size]) / (5*I0.nominal_value) # 5s Messzeit
x = np.linspace(0, 1.5, 10000)
reflectivity_corrected = reflectivity.copy()    
# reflectivity_corrected[(thetaReflect < winkel_experiment) & (thetaReflect > 0)] /= np.sin((winkel_experiment))/np.sin((thetaReflect[(thetaReflect < winkel_experiment) & (thetaReflect > 0)]))
reflectivity_corrected /= geom_factor
reflectivity_corrected[0] = 0

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


minima = []
for i in range(6,len(thetaReflect)):
    if (thetaReflect[i] > 0.2) and (thetaReflect[i] < 1) and all(reflectivity_corrected[i] < reflectivity_corrected[i+k] for k in range(1, 5)) and all(reflectivity_corrected[i] < reflectivity_corrected[i-k] for k in range(1, 5)):
        minima.append(i)


plt.errorbar(thetaReflect[-len(thetaReflect)+1:], reflectivity[-len(reflectivity)+1:], label="Reflect ohne Korrekturfaktor", color="r")
plt.errorbar(thetaReflect[-len(thetaReflect)+1:], reflectivity_corrected[-len(reflectivity_corrected)+1:], label="Reflektivität mit Korrekturfaktor", color="darkred", linestyle="dashed", alpha=0.7)
plt.plot(x, fresnel_ideal(x), label="Fresnelreflektivität", color="b")
plt.plot(thetaReflect[minima], reflectivity_corrected[minima], "x", color="k", label="Minima")
plt.vlines(critical_angle, 0, np.max(reflectivity_corrected), color="gray", linestyle="dashed", label=r"Kritischer Winkel $\alpha_c$")

plt.xlim(0,1.5)
# plt.ylim(np.max(reflectivity_corrected))
plt.yscale("log")
plt.xlabel(r"$\theta\,/\, \text{DEG}$")
plt.ylabel(r"$R$")
plt.grid()
plt.legend(loc="best")
plt.savefig("build/Reflectivity.pdf")
plt.clf()


diff = np.diff(thetaReflect[minima])
lambda_ka = 1.54e-10 # Angström -> Wellenlaenge der Roentgenstrahlung bei der Kalpha Linie
delta_a = ufloat(np.mean(diff), np.std(diff))
d = lambda_ka/(2*delta_a*np.pi/180)

print("-----------------Schichtdicke-----------------------")
print(f"Delta alpha:    {delta_a:.2e}")
print(f"Schichtdicke:   {d:.2e}")
print("---------------------------------------------------")

#######################################################
### Parratt-Algorithmus
#######################################################

k = 2*np.pi/lambda_ka   # Wellenvektor
n1 = 1
d1 = 0

theta_min = 0.2     #Startwert Fit
theta_max = 0.8     #Endwert Fit
x = np.linspace(0,1.5,1000)


d_poly = 1.7e-6 # 1. Schicht Polysterol
d_Sili = 8.0e-6 # 2. Schicht Silizium
b_poly = d_poly / 200
b_sili = d_Sili / 400
d = 9.0e-8
sigma_poly = 5.2e-10
sigma_sili = 7.8e-10

start_params = [d_poly, d_Sili, b_poly, b_sili, d, sigma_poly, sigma_sili]
boundaries = ([1e-7, 1e-7, 1e-10, 1e-10, 1e-9, 5e-12, 5e-12], 
                [5e-5, 5e-4, 1e-4, 1e-4, 1e-7, 1e-9, 1e-9]) # Limits der Parameter

delta_Sili = ufloat(params[1], errors[1])
delta_poly = ufloat(params[0], errors[0])
crit_winkel_sili = unp.sqrt(2*delta_Sili)#*np.pi/180
crit_winkel_poly = unp.sqrt(2*delta_poly)#*np.pi/180


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


plt.plot(x, parratt(x, *start_params), label="Parratt-Algorithmus", color="b")
plt.plot(x, fresnel_ideal(x), label="Fresnelreflektivität", color="gray", linestyle="dashed", alpha=.8)
plt.errorbar(thetaReflect[-len(thetaReflect)+1:], reflectivity_corrected[-len(reflectivity_corrected)+1:]*.1, label="Reflektivität mit Korrekturfaktor", color="red")


# err = np.sqrt(np.diag(covariance))

plt.xlabel(r"$\theta\,/\, \text{DEG}$")
plt.ylabel(r"$R$")
plt.xlim(0,1.5)
# plt.ylim(1.8*10**(-6),1.5*10**(1))
plt.grid()
plt.legend(loc="best")
plt.yscale("log")
plt.savefig("build/Parratt.pdf")
# plt.clf()
fit_params, covariance = curve_fit(parratt, thetaReflect, reflectivity_corrected, p0=start_params)#, bounds=boundaries)


print("-----------------Parratt-Algorithmus-----------------------")
print(f"delta Silizium:                 {fit_params[0]:.2e}")
print(f"delta Poly:                     {fit_params[1]:.2e}")
print(f"b Silizium:                     {fit_params[2]:.2e}")
print(f"b Poly:                         {fit_params[3]:.2e}")
print(f"d:                              {fit_params[4]:.2e}")
print(f"sigma Silizium:                 {fit_params[5]:.2e}")
print(f"sigma Polysterol:               {fit_params[6]:.2e}")
print(f"Kritischer Winkel Silizium:     {crit_winkel_sili:.2}")
print(f"Kritischer Winkel Polysterol:   {crit_winkel_poly:.2}")


'''
--------------Parameter der Gaußfunktion---------------
alpha0  : 2.6767e-03 +/- 4.6026e-04
sigma   : 3.6969e-02 +/- 4.7823e-04
I0      : (3.4247+/-0.0411)e+04
B       : 5.1979e+03 +/- 9.1778e+02
FWHM    : 0.0871+/-0.0011
-------------------------------------------------------
Strahlbreite: 0.2 mm
-----------------Geometriewinkel-----------------------
Experiment: 0.4°
Theorie:    0.57°
-----------------Schichtdicke-----------------------
Delta alpha:    (5.12+/-0.49)e-02
Schichtdicke:   (8.62+/-0.82)e-08
-----------------Parratt-Algorithmus-----------------------
delta Silizium:                 1.67e-06
delta Poly:                     8.89e-06
b Silizium:                     -2.41e-08
b Poly:                         5.05e-10
d:                              1.21e-07
sigma Silizium:                 -7.94e-08
sigma Polysterol:               3.59e-11
Kritischer Winkel Silizium:     0.27+/-0.00
Kritischer Winkel Polysterol:   0.073+/-0.006
'''