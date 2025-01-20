import matplotlib.pyplot as plt
import numpy as np

# f, u1, u2, phase = np.genfromtxt("data/Invert_100k.txt", delimiter=";", skip_header=1, unpack=True)
# u_quot = u1 / (u2/1000)
# plt.plot(f, u_quot, ".", label="100 k$\Omega$")

# f, u1, u2, phase = np.genfromtxt("data/Invert_15k.txt", delimiter=";", skip_header=1, unpack=True)
# u_quot = u1 / (u2/1000)
# plt.plot(f, u_quot, ".", label="15 k$\Omega$")

# f, u1, u2, phase = np.genfromtxt("data/Invert_47k.txt", delimiter=";", skip_header=1, unpack=True)
# u_quot = u1 / (u2/1000)
# plt.plot(f, u_quot, ".", label="47 k$\Omega$")

f, u1, u2, phase = np.genfromtxt("data/Invert_220k.txt", delimiter=";", skip_header=1, unpack=True)
u_quot = u1 / (u2/1000)
plt.plot(f, u_quot, ".", label="220 k$\Omega$")


# plt.plot(f, phase, "r.", label="Phase")

plt.xscale("log")
plt.yscale("log")
plt.legend(loc='best')
plt.show()
