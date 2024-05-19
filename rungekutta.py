import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Simu:
    def __init__(self, m1, m2, u, tmax=10, N=1000):
        self.eps = 10E-2
        self.t = np.linspace(0, tmax, N)
        self.m1, self.m2 = m1, m2
        self.g = 9.81
        self.l = .2
        self.mu2 = m2 / (m1 + m2)
        self.w0 = np.sqrt(self.g / self.l)
        self.u_ = u

    def pend(self, y, t_, w):
        du = np.gradient(self.u_(w, self.t))
        d2u = np.gradient(du)
        def up_(w, t_):
            i = np.argwhere(np.abs(self.t - t_) < self.eps)
            return du[i][0][0]
        def upp_(w, t_):
            i = np.argwhere(np.abs(self.t - t_) < self.eps)
            return d2u[i][0][0]
        u, up, upp = self.u_(w, t_), up_(w, self.t), upp_(w, self.t)
        return np.array([y[1], (self.mu2 * upp * (1+np.cos(u)) - self.w0**2 * self.mu2 * np.sin(y[0] - u) + 2 * self.mu2 * y[1] * up * np.sin(u) - self.mu2 * up**2 * np.sin(u) - 2 * self.w0**2 * np.sin(y[0]) ) / (2 + self.mu2 * (1 + np.cos(u)))])

    def simu(self, y0, w):
        n = len(self.t)
        y = np.zeros((n, len(y0)))
        y[0] = y0
        for i in range(n - 1):
            h = self.t[i+1] - self.t[i]
            k1 = self.pend(y[i], self.t[i], w)
            k2 = self.pend(y[i] + k1 * h / 2., self.t[i] + h / 2., w)
            k3 = self.pend(y[i] + k2 * h / 2., self.t[i] + h / 2., w)
            k4 = self.pend(y[i] + k3 * h, self.t[i] + h, w)
            y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
        return self.t, y
    
    def ampl(self, wmax=10, N=100):
        ws = np.linspace(0, wmax, N)
        A = []
        Wp = []

        for w_ in tqdm(ws):
            sol = self.simu([0,0], w_)[-1]
            t1 = sol[:,0]
            t1p = sol[:,1]
            A.append(np.amax(t1))
            Wp.append(np.amax(t1p))
        plt.plot(ws, A)
        #plt.plot(ws, Wp)
