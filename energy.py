import numpy as np
import matplotlib.pyplot as plt
from pendulum_detect.main import *

g, l = 9.81, 0.2
m1, m2 = 0.2,0.2


import scienceplots
plt.style.use('science')

t1, t2 = np.linspace(-np.pi/2, np.pi/2, 100), np.linspace(-np.pi/2, np.pi, 100)

V = []
for a in t1:
    temp = []
    for b in t2:
        temp.append((-m1*g*l*np.cos(a) - m2*g*l*(np.cos(a) + np.cos(b)))/(m1*g*l + 2*m2*g*l))
    V.append(temp)


plt.pcolormesh(t1, t2, V, cmap=plt.colormaps.get_cmap('gist_heat'), shading="gouraud")
plt.colorbar(label=r"$E_p(\theta_1,\theta_2)$ normalisée")

plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")

plt.savefig("ep.png", dpi=300)


def f(a, b):
    return (-m1*g*l*np.cos(a) - m2*g*l*(np.cos(a) + np.cos(b)))/(m1*g*l + 2*m2*g*l)

X, Y = np.meshgrid(t1, t2)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, f(X,Y), rstride=1, cstride=1,
                cmap='gist_heat', edgecolor='none')

plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")

plt.savefig("saut_pot.png", dpi=300)

"""
def tp(t, dt):
    tp_ = []
    for i in range(len(t)-1):
        tp_.append((t[i+1] - t[i])/dt)
    return np.array(tp_)

def Ec(t1, t2):
    return m1*np.power(tp(t1, 0.1), 2)*l**2/2 + m2 * l**2 * (np.power(tp(t1, 0.1),2) + np.power(tp(t2, 0.1),2) + 2*tp(t1, 0.1)*tp(t2, 0.1)*np.cos(t2[:-1]-t1[:-1]))

def Ep(t1, t2):
    return - m1*g*l*np.cos(t1) - m2*g*l*(np.cos(t1) + np.cos(t2))

res = get_from_file("IMG_2208.txt")

import scienceplots
plt.style.use('science')

angles = ResultHandling.getAngles(res, (169,160), (720,480))
s = ResultHandling.detect_start(angles, 0.1)

a, b = ResultHandling.ft(angles[1][313:], 0.1)


plt.plot(angles[1][313:], label=r"$\theta_2$")
plt.plot(angles[0][313:], label=r"$\theta_1$")

plt.xlabel("Temps (x100s)")
plt.ylabel("Angle (rad)")
plt.legend()

ec = Ec(np.array(angles[0][313:]), np.array(angles[1][313:]))
ep = Ep(np.array(angles[0][313:]), np.array(angles[1][313:]))

em = ec + ep[:-1]

dem = tp(em, 0.1)
t1p = tp(np.array(angles[0][313:]), 0.1)[:-1]
t2p = tp(np.array(angles[1][313:]), 0.1)[:-1]

def absurd_value(arr, ar1, ar2, val):
    for i in range(len(arr)):
        if abs(arr[i]) > val:
            return np.delete(arr, i), np.delete(ar1, i), np.delete(ar2, i)
    return arr, ar1, ar2

## Avoiding absurd value
for i in range(len(dem)):
    dem, t1p, t2p = absurd_value(dem, t1p, t2p, 10)


plt.figure(figsize=(4,2))

plt.plot(t1p + t2p, -dem, "x", color="k", label="Mesures")

a, b = np.polyfit(t1p + t2p, dem, 1)

X = np.linspace(min(t1p + t2p), max(t1p + t2p), 100)
BestFit = a*X + b

plt.plot(X, -BestFit, "r", label="Best Fit")
plt.xlabel(r"$\dot{\theta}_1 + \dot{\theta}_2$")
plt.ylabel(r"$\frac{d(E_c + E_p)}{dt}$")
plt.legend()
plt.xlim([-1,1])
plt.ylim([-1,1])

print(a, b)
plt.savefig("tem.png", dpi=300)

fig = plt.figure()
ax = plt.axes(projection='3d')


def f(a, b):
    return (-m1*g*l*np.cos(a) - m2*g*l*(np.cos(a) + np.cos(b)))/(m1*g*l + 2*m2*g*l)

t1, t2 = np.linspace(-np.pi/2, np.pi/2, 100), np.linspace(-np.pi/2, np.pi, 100)
X, Y = np.meshgrid(t1, t2)

ax.plot_wireframe(X, Y, f(X,Y), color='lightgray')
ax.plot3D(angles[0][313:], angles[1][313:], ep/max(abs(ep)), "o", color="k")

ax.set_ylim([-np.pi/2, 2*np.pi/2])
ax.set_xlim([-np.pi/2, np.pi/2])

plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
"""

plt.show()