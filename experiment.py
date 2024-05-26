### Parametric double pendulum experiment

import serial
import numpy as np
import matplotlib.pyplot as plt

#com = serial.Serial("COM1", baudrate=9600, timeout=0)

def u(A, w, t):
    return A * (1 + np.sin(w * t))

w0 = 7.9
t = np.linspace(0, 2*np.pi / w0, 100)
u_ = u(20, w0, t)

print('{' + ",".join(str(_) for _ in u_) + '}')

plt.plot(t, u_, "o", ms=1)
plt.xlabel("Temps (s)")
plt.ylabel(r"$u(t)$ (deg)")
plt.show()

