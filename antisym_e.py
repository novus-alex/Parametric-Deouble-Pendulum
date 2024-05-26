import numpy as np
import matplotlib.pyplot as plt
from pendulum_detect.main import *
import scienceplots
plt.style.use('science')

g, l = 9.81, 0.2
m1, m2 = 0.2,0.2

res = get_from_file("IMG_2211.txt")
angles = ResultHandling.getAngles(res, (485,617), (720,480))

def tp(t, dt):
    tp_ = []
    for i in range(len(t)-1):
        tp_.append((t[i+1] - t[i])/dt)
    return np.array(tp_)

def Ec(t1, t2):
    return m1*np.power(tp(t1, 0.1), 2)*l**2/2 + m2 * l**2 * (np.power(tp(t1, 0.1),2) + np.power(tp(t2, 0.1),2) + 2*tp(t1, 0.1)*tp(t2, 0.1)*np.sin(t2[:-1]-t1[:-1]))

def Ep(t1, t2):
    return - m1*g*l*np.cos(t1) - m2*g*l*(np.cos(t1) + np.cos(t2))

ec = Ec(np.array(angles[0][0:]), np.array(angles[1][0:]))
ep = Ep(np.array(angles[0][0:]), np.array(angles[1][0:]))

def f(a, b):
    return (-m1*g*l*np.cos(a) - m2*g*l*(np.cos(a) + np.cos(b)))/(m1*g*l + 2*m2*g*l)

t1, t2 = np.linspace(-1, 1, 100), np.linspace(-2*np.pi, 2*np.pi, 100)
X, Y = np.meshgrid(t1, t2)

plt.plot(t1, f(0,t1))
plt.errorbar(angles[1], ep/max(abs(ep)), 0.01, 0, fmt="o", color="k", lw=1, ms=3, capsize=3, zorder = 2, label="Mesures")
plt.xlabel(r"$\theta_2$")
plt.ylabel(r"E_p")
plt.legend()
plt.show()
#plt.savefig("sym_ep.png", dpi=300)