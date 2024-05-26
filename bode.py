import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from rungekutta_dp import *

plt.style.use(["science", "no-latex"])

f_ = ['7_4.txt', '7_6.txt', '7_8.txt', '8.txt']
w = [7.4, 7.6, 7.8, 8]
A = []

for name in f_:
    t, T = [], []
    with open(name, "r") as f:
        for line in f.readlines():
            data = line.split(";")
            t.append(float(data[1])); T.append(float(data[0]))
    A.append(np.amax(np.abs(T)))

S = Simu(1.7, 1, u_, tmax=max(t))
S.ampl(N=100)

plt.plot(w,A, "o")
plt.show()