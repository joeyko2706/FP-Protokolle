import matplotlib.pyplot as plt          # type: ignore
import numpy as np                       # type: ignore
from uncertainties import ufloat, unumpy # type: ignore
from scipy.optimize import curve_fit     # type: ignore
from scipy import constants              # type: ignore

theta, max, min = np.genfromtxt('data/contrast.csv', delimiter=';', unpack=True)
theta = np.unique(theta)

pressure, meas1, meas2, meas3, counts, std = np.genfromtxt('data/refrective_air.csv', delimiter=';', unpack=True, skip_header=1)
n_glass = ufloat(np.mean([31, 34, 34, 33, 35]), np.std([31, 34, 34, 33, 35]))

k = 0
maximum_avg = []
minimum_avg = []
maximum_std = []
minimum_std = []
for i in range(len(max)):
    maximum_avg.append(np.mean(max[k] + max[k+1] + max[k+2]))
    minimum_avg.append(np.mean(min[k] + min[k+1] + min[k+2]))
    maximum_std.append(np.std([max[k], max[k+1], max[k+2]]))
    minimum_std.append(np.std([min[k], min[k+1], min[k+2]]))

    k += 3
    if k == len(max):
        break


maximum = unumpy.uarray(maximum_avg, maximum_std)
minimum = unumpy.uarray(minimum_avg, minimum_std)


def contrast(i_max:ufloat, i_min:ufloat)->ufloat:
    return (i_max-i_min)/(i_max+i_min)


def theory(x, k_0, delta):
    phi = np.radians(x)
    return 2*k_0*np.abs(np.sin(phi-delta)*np.cos(phi-delta))


measured_contrast = contrast(maximum, minimum)
params, covariance = curve_fit(theory, theta[:-1], unumpy.nominal_values(measured_contrast), p0=[0,1], sigma=unumpy.std_devs(measured_contrast))

x = np.linspace(0, 180, 1000)   # 1000 points between 0 and 180 in degrees (important)
plt.errorbar(theta[:-1], unumpy.nominal_values(measured_contrast), yerr=unumpy.std_devs(measured_contrast), fmt='bx', label='Messwerte')
plt.plot(x, theory(x, *params), color="r", label='Fit')
plt.grid()
plt.xlabel(r'$\vartheta \,/\, ^\circ$')
plt.ylabel(r'contrast')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('build/contrast.pdf')
# plt.show()
plt.clf()

print('--------Kontrastbestimmung--------')
print(f'k_0 = {ufloat(params[0], np.sqrt(covariance[0,0]))}')
print(f'delta = {ufloat(params[1], np.sqrt(covariance[1,1]))}')

k_0 = ufloat(params[0], np.sqrt(covariance[0,0]))
delta = ufloat(params[1], np.sqrt(covariance[1,1]))

R = constants.R
T = 20.2 + 273.15   # in Kelvin
lamda = 632.99e-9   # wavelength of the laser (meter)
L = ufloat(0.1, 0.1e-3)
d = 1e-3            # thickness of the glass plate (meter)
n_glas_theory = 1.4570  #https://refractiveindex.info
m_bar = ufloat(33.4, 1.4)  # Mittelwert und Standardabweichung der Counts der Messung vom Glas


def n_air_exp(M):
    return M*lamda/(L) + 1


def refraction_index(p, a, b, T=T, R=R):
    return 3/2 * p/(T*R) * a + b


def refraction_index_glass(M, theta):
    theta = np.radians(theta)
    return 1/(1-(M*lamda)/(2*d*theta**2))


counting_rate = unumpy.uarray(counts, std)
n = n_air_exp(counting_rate)
popt, pcov = curve_fit(refraction_index, pressure, unumpy.nominal_values(n), sigma=unumpy.std_devs(n))#, p0=[.042, 0])
err = np.sqrt(np.diag(pcov))

x = np.linspace(50, 1000, 1000)
plt.errorbar(pressure, unumpy.nominal_values(n), yerr=unumpy.std_devs(n), fmt='bx', label='Messwerte')
plt.plot(x, refraction_index(x, *popt), color="r", label='Fit')
plt.grid()
# plt.xlabel(r'$p \,/\, \si{\milli\bar}$')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-5, -5), useMathText=True)
plt.ylabel(r'n')
plt.legend()
plt.tight_layout()
plt.savefig('build/refraction_index.pdf')
# plt.show()

n_air_exp = 1- refraction_index(0, ufloat(popt[0], np.sqrt(pcov[0,0])), ufloat(popt[1], np.sqrt(pcov[1,1])))
n_glass_exp = refraction_index_glass(m_bar, 10)

print('--------Brechungsindexbestimmung--------')
print(f'n_glass_exp = {n_glass_exp}')
print(f'a = {ufloat(popt[0], np.sqrt(pcov[0,0]))}')
print(f'b = 1 - {1-ufloat(popt[1], np.sqrt(pcov[1,1]))}')
print(f'n_air_exp = 1 - {n_air_exp}')
print(f'n_standard_air = {refraction_index(1013, ufloat(popt[0], np.sqrt(pcov[0,0])), ufloat(popt[1], np.sqrt(pcov[1,1])), T=288.15)}')
print('--------Relative Abweichungen--------')
print(f'Kontrast: {100*(k_0-1)/1:.2f}%')
print(f'Brechungsindex Glas: {100*(n_glass_exp/n_glas_theory-1):.2f}%')
print(f'Brechungsindex Luft, normal: {100*(refraction_index(0, *popt)-1)/1.00027653:.8f}%')
print(f'Brechungsindex Luft, standard: {100*(1-refraction_index(1013, ufloat(popt[0], np.sqrt(pcov[0,0])), ufloat(popt[1], np.sqrt(pcov[1,1])), T=288.15)/1.00027653)}')


# '''Print the data to use in Latex table'''

# print('--------Kontrast--------')
# for i in range(len(theta[:-1])):
#     print(f'{theta[i]} & ${maximum[i]:.2f}$ & ${minimum[i]:.2f}$ & ${measured_contrast[i]:.2f}$ \\\\')


# print('--------Brechungsindex--------')
# for i in range(len(pressure)):
#     print(f'{pressure[i]} & ${counting_rate[i]}$\\\\')

# print('\n', 'n_glass = ', n_glass)