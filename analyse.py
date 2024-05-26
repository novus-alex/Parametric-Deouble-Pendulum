import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from rungekutta_dp import *

plt.style.use(["science", "no-latex"])

t, T = [], []
with open("8.txt", "r") as f:
    for line in f.readlines():
        data = line.split(";")
        t.append(float(data[1])); T.append(float(data[0]))

def ft(angle, Te):
    from numpy.fft import fft, fftfreq

    angle = np.array(angle)
    X = fft(angle)
    freq = fftfreq(angle.size, d=Te)
    N = angle.size
    X_abs = np.abs(X[1:N//2])
    X_norm = X_abs*2.0/N
    freq_pos = freq[1:N//2]

    return X_norm, freq_pos


X, F = ft(T, 1/(max(t)))
S = Simu(1.7, 1, u_, tmax=max(t))
t_, y = S.simu([0,0], 7.8)

Xp, Fp = ft(y[:,0], 1/max(t_))


fig = plt.figure(figsize=(6,3))

"""
plt.plot(Fp, Xp)
plt.vlines(6.8/(2*np.pi), 0, 0.2, label=r"$f_0$", color="g")
plt.legend()
plt.plot(F, X)
plt.xlabel("Frequence (Hz)")
plt.ylabel("Amplitude")

"""
plt.plot(t, T)
plt.xlabel("Temps (s)")
plt.ylabel(r"$\theta_1$")

plt.savefig("8_meas.png", dpi=300)
#plt.show()