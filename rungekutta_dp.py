import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

"""
def u_(w, t):
    return a*(1+np.sin(w*t))
def up_(w, t):
    return -a*w*np.cos(w*t)
def upp_(w, t):
    return -a*w**2 * np.sin(w*t)
"""


def u_(w, t):
    return 0.8 / (1 + np.exp(-30*np.cos(w*t)))


"""
def w(t):
    return w0 * (1 - 1 / (1 + np.exp(-10 * (t-5))))

sol = simu(pend, [0,0], t, w)

def pend_anim():
    sol = simu(pend, [0,0], t, 7.9)
    x1 = np.sin(sol[:, 0])
    y1 = -np.cos(sol[:, 0])

    x2 = np.sin(sol[:, 0] - u_(7.9, t)) + x1
    y2 = -np.cos(sol[:, 0] - u_(7.9, t)) + y1

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2,2), ylim=(-2.5, 1.))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]

        history_x = x2[:i]
        history_y = y2[:i]

        line.set_data(thisx, thisy)
        trace.set_data(history_x, history_y)
        time_text.set_text(time_template % (i*max(t)/len(t)))
        return line, trace, time_text


    ani = animation.FuncAnimation(
        fig, animate, len(sol[:,0]), interval=max(t)/len(t)*1000, blit=True)
    plt.show()

theta_1 = sol[:,0]
theta_2 = theta_1 - u_(1, t)

t1p = sol[:,1]
t2p = t1p - up_(1, t)

Ep = - g * np.cos(theta_1) - g * (np.cos(theta_2) + np.cos(theta_1))
Ec = t1p**2 / 2 + (t1p **2 + t2p**2 + 2*t1p * t2p * np.sin(theta_2 - theta_1)) / 2
"""
