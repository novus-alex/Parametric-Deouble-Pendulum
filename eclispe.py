import numpy as np
import matplotlib.pyplot as plt

G = 6.67E-11
Ms = 1.989E30
Mt = 5.972E24
Ml = 7.342E22

Dts = 149597870.7
Dtl = 384400

class Body:
    def __init__(self, mass, c):
        self.mass = mass
        self.c = np.array(list(c), dtype=np.float64)
        self.v = np.array([0,0], dtype=np.float64)

    def set_v(self, v):
        self.v = np.array(v, dtype=np.float64)

def dist(a,b):
    return np.sqrt(np.sum((a-b)**2))

def compute(B, dt):
    for b in B:
        F = np.array([0,0], dtype=np.float64)
        for bp in B:
            if bp != b:
                d = dist(b.c,bp.c) 
                vF = bp.c - b.c
                vF *= -G * bp.mass / d**3
                F += vF
        b.v += F * dt
        b.c += b.v * dt

Sun = Body(Ms, (0,0))
Earth = Body(Mt, (Dts,0))
Moon = Body(Ml, (Dts,Dtl))

Earth.set_v([0,Dts*np.sqrt(G*Ms / Dts**3)])

Bs = [Sun, Earth]
X, Y = [], []
for i in range(100):
    compute(Bs, 0.1)
    X.append(Earth.c[0]); Y.append(Earth.c[1])


plt.plot(X,Y)
plt.show()