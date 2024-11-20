import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from scipy.optimize import curve_fit


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
plt.vlines(vline , 0, 400000, color = "gray", alpha=.7, label = f"FWHM: {FWHM.nominal_value:.3f}({FWHM.std_dev:.3f})°")

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
upperLimit = 35
lowerLimit = 15
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
winkel_experiment = np.deg2rad(theta[upperLimit]- theta[lowerLimit])
print("-----------------Geometriewinkel-----------------------")
print(f"Experiment: {winkel_experiment:.2}°")
print(f"Theorie:    {winkel_theorie:.2}°")
print("-------------------------------------------------------")

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

critical_angle = 0.223 # deg
reflectivity = (hitsReflect - hitsDiffuser[thetaReflect.size]) / (5*I0.nominal_value) # 5s Messzeit
x = np.linspace(critical_angle, 1.5, 10000)
x2 = np.linspace(0,critical_angle,2)
reflectivity_corrected = reflectivity.copy()    
reflectivity_corrected[(thetaReflect < winkel_experiment) & (thetaReflect > 0)] *= np.sin(np.deg2rad(winkel_experiment))/np.sin(np.deg2rad(thetaReflect[(thetaReflect < winkel_experiment) & (thetaReflect > 0)]))


def fresnel(alpha):
    return(critical_angle/(2*alpha))**4


minima = []
for i in range(6,len(thetaReflect)):
    if (thetaReflect[i] > 0.2) and (thetaReflect[i] < 1) and all(reflectivity_corrected[i] < reflectivity_corrected[i+k] for k in range(1, 5)) and all(reflectivity_corrected[i] < reflectivity_corrected[i-k] for k in range(1, 5)):
        minima.append(i)


plt.errorbar(thetaReflect, reflectivity, label="Reflect ohne Korrekturfaktor", color="r")
plt.errorbar(thetaReflect, reflectivity_corrected, label="Reflect mit Korrekturfaktor", color="darkred", linestyle="dashed", alpha=0.7)
plt.plot(x, fresnel(x), label="Fresnelreflektivität", color="b")
plt.plot(x2, 2*[fresnel(critical_angle)], color="b")
plt.plot(thetaReflect[minima], reflectivity[minima], "x", color="k", label="Minima")
plt.vlines(critical_angle, 0, 10**2, color="gray", linestyle="dashed", label=r"Kritischer Winkel $\alpha_c$")

plt.xlim(0,1.5)
plt.ylim(10**(-5),10**(2))
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
d_poly = 4.2e-6 # 1. Schicht des Poiysterols
d_Sili = 1.6e-5 # 1. Schicht des Siliziums
b_poly = 2.7e-8
b_sili = 9.8e-7
d = 8.2e-8  
sigma_poly = 4e-10
sigma_sili = 3e-10
theta_min = 0.2     #Startwert Fit
theta_max = 0.8     #Endwert Fit
x = np.linspace(0,1.5,1000)

start_params = [d_poly, d_Sili, b_poly, b_sili, d, sigma_poly, sigma_sili]
boundaries = ([1e-7, 1e-7, 1e-10, 1e-10, 1e-9, 5e-12, 5e-12], [5e-5, 5e-5, 1e-6, 1e-6, 1e-7, 1e-9, 1e-9]) # Limits der Parameter
err = np.zeros(len(start_params))

delta_Sili = ufloat(params[1], errors[1])
delta_poly = ufloat(params[0], errors[0])
crit_winkel_sili = unp.sqrt(2*delta_Sili)*np.pi/180
crit_winkel_poly = unp.sqrt(2*delta_poly)*np.pi/180


def parratt(alpha, delta2, delta3, b2, b3, d2, sigma1, sigma2): # Parratt-Algorithmus
    n2, n3 = 1.0 - delta2 - 1j*b2, 1.0 - delta3 - 1j*b3         # Brechungsindizes
    alpha = np.deg2rad(alpha)                                   # Umrechnung in Bogenmaß
    kz1, kz2, kz3 = k * np.sqrt(n1**2 - np.cos(alpha)**2), k * np.sqrt(n2**2 - np.cos(alpha)**2), k * np.sqrt(n3**2 - np.cos(alpha)**2)         # Wellenvektoren k_z_j
    r12, r23 = ((kz1 - kz2)/(kz1 + kz2)) * np.exp(-2 * kz1 * kz2 * sigma1**2), ((kz2 - kz3)/(kz2 + kz3)) * np.exp(-2 * kz2 * kz3 * sigma2**2)   # Reflektivitäten
    return np.abs((r12 + np.exp(-2j * kz2 * d2) * r23) / (1 + r12 * np.exp(-2j * kz2 * d2) * r23))**2                                           # Gesamtreflektivität                                         


plt.plot(x, parratt(x, *start_params), label="Parratt-Algorithmus", color="b", alpha=.8)
plt.plot(thetaReflect, reflectivity_corrected, label="Reflektivitäten (mit Korrekturfaktor)", color="r", alpha=.8)


x = np.linspace(critical_angle, 1.5, 10000)
x2 = np.linspace(0,critical_angle,2)
plt.plot(x, fresnel(x), label="Fresnelreflektivität", color="gray", linestyle="dashed", alpha=.8)
plt.plot(x2, 2*[fresnel(critical_angle)], color="gray", linestyle="dashed", alpha=.8)


start_params, covariance = curve_fit(parratt, thetaReflect, reflectivity_corrected, p0=start_params, bounds=boundaries)
# err = np.sqrt(np.diag(covariance))

plt.xlabel(r"$\theta\,/\, \text{DEG}$")
plt.ylabel(r"$R$")
plt.xlim(0,1.5)
plt.grid()
plt.legend(loc="best")
plt.yscale("log")
plt.savefig("build/Parratt.pdf")


print("-----------------Parratt-Algorithmus-----------------------")
print(f"delta Silizium:                 {start_params[0]:.2e}")
print(f"delta Poly:                     {start_params[1]:.2e}")
print(f"b Silizium:                     {start_params[2]:.2e}")
print(f"b Poly:                         {start_params[3]:.2e}")
print(f"d:                              {start_params[4]:.2e}")
print(f"sigma Silizium:                 {start_params[5]:.2e}")
print(f"sigma Polysterol:               {start_params[6]:.2e}")
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
Experiment: 0.014°
Theorie:    0.01°
-------------------------------------------------------
-----------------Schichtdicke-----------------------
Delta alpha:    (5.12+/-0.45)e-02
Schichtdicke:   (8.62+/-0.75)e-08
---------------------------------------------------
-----------------Parratt-Algorithmus-----------------------
delta Silizium:                 9.53e-06
delta Poly:                     8.44e-06
b Silizium:                     1.00e-10
b Poly:                         3.74e-07
d:                              2.40e-08
sigma Silizium:                 1.40e-11
sigma Polysterol:               2.90e-10
Kritischer Winkel Silizium:     0.0047+/-0.0000
Kritischer Winkel Polysterol:   0.0013+/-0.0001
'''