""" Alexandre Hachet 2023

Script that read a video from a simple/double pendulum experiment and extract all the points 

HSV detection points: 85 and 105
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import *

def get_range(wl):
    return [(np.array([w-5, 100, 140]), np.array([w+5, 255, 255])) for w in wl]

def detect_func(thresh, offset):
    coords = np.column_stack(np.where(thresh > 0))
    mean_co = coords.sum(axis=0) / len(coords)
    return offset[0] - mean_co[0], mean_co[1]

def detect(filename, wl, detect_f, write_to_file=False, show_masks=False):
    cap = cv2.VideoCapture(filename)
    vid_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    wl_range = get_range(wl)

    det = {}
    for i in range(len(wl)):
        det[i] = []
    i = 0
    
    while (1):
        _, frame = cap.read()
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        except:
            break

        results = []
        for wl_ in wl_range:
            mask = cv2.inRange(hsv, wl_[0], wl_[1])
            results.append(cv2.bitwise_and(frame, frame, mask = mask))

        if i > 0:
            threshs = []
            for j in range(len(results)):
                thresh = cv2.threshold(results[j], 230, 240, cv2.THRESH_BINARY)[1]
                det[j].append(detect_f(thresh, frame.shape))
                threshs.append(cv2.dilate(thresh, None, iterations=2))

            if show_masks:
                #plt.imshow(frame)
                #plt.show()
                cv2.imshow("r1", threshs[0])
                cv2.imshow("r2", threshs[1])
                cv2.imshow("vid", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        i += 1
        print(f"Progress: {i} of {int(vid_len)}", end="\r")

    if show_masks: 
        cv2.destroyAllWindows()
        cap.release()

    if write_to_file:
        with open(filename.split(".")[0] + ".txt", "w") as f:
            for j in range(len(det[0])):
                f.write(f"{det[0][j][0]};{det[0][j][1]}:{det[1][j][0]};{det[1][j][1]}\n")
    return det

def get_from_file(filename):
    res = [[], []]
    with open(filename, "r") as f:
        for line in f.readlines():
            data = line.split(":")
            for i in range(len(data)):
                s_data = data[i].split(";")
                res[i].append((float(s_data[0]), float(s_data[1])))
    return res


class Tools:
    def isPiMultiple(n, eps):
        i = 0
        while abs(abs(n) - i*pi) > eps:
            i += 1
        return i


class ResultHandling:
    def animation(res):
        import matplotlib.animation as animation

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(0, 480), ylim=(0, 720))
        ax.set_aspect('equal')
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def animate(i):
            thisx = [169, res[0][i][1], res[1][i][1]]
            thisy = [720 - 160, res[0][i][0], res[1][i][0]]
            
            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i*0.1))
            return line, time_text

        ani = animation.FuncAnimation(fig, animate, range(1, len(res[0])),
                                    interval=0.1*1000, blit=True, init_func=init)
        plt.show()

    def getAngles(res, origin, shape):
        """
        res[0][i][1] = x1 et res[1][i][1] = x2
        res[0][i][0] = y1 et res[1][i][0] = y2
        """
        #origin = (shape[0] - origin[0], origin[1])

        thetas = [[] for i in range(len(res))]
        for j in range(len(res[0])):
            thetas[0].append(atan((origin[1] - res[0][j][1])/(- origin[0] + res[0][j][0])))
        for i in range(1, len(res)):
            p = 0
            for j in range(len(res[i])):
                if res[i-1][j][0] - res[i][j][0] < 0:
                #if abs(abs(atan((res[i-1][j][1] - res[i][j][1])/(res[i-1][j][0] - res[i][j][0]))) - pi/2) < 0.1:
                    #p = atan((res[i-1][j][1] - res[i][j][1])/(res[i-1][j][0] - res[i][j][0])) // (pi/2)
                    p = 1
                thetas[i].append(pi/2 * p + atan((res[i-1][j][1] - res[i][j][1])/(res[i-1][j][0] - res[i][j][0])))
        return thetas
    
    def getVelocity(angles, dt):
        w = [[] for i in range(len(angles))]
        for i in range(len(angles)):
            for j in range(len(angles[i])-1):
                w[i].append((angles[i][j+1]-angles[i][j])/dt)
        return w
    
    def detect_start(angles, eps):
        the, i = angles[0], 1
        while abs(the[i] - the[i-1]) < eps:
            i += 1
        return i
    
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


#res = detect("IMG_2211.MOV", [85, 105], detect_func, write_to_file=True, show_masks=True)
"""
res = get_from_file("IMG_2211.txt")

#plt.plot([atan((617 - res[0][_][1])/(485 - res[0][_][0])) for _ in range(len(res[0]))])
#plt.plot([485 - res[0][_][0] for _ in range(len(res[0]))])


import scienceplots
plt.style.use('science')

#angles = ResultHandling.getAngles(res, (169,160), (720,480))
#s = ResultHandling.detect_start(angles, 0.1)
angles = ResultHandling.getAngles(res, (485,617), (720,480))

a, b = ResultHandling.ft(angles[1][313:], 0.1)

plt.plot(angles[1][50:], label=r"$\theta_2$")
plt.plot(angles[0][50:], label=r"$\theta_1$")

plt.xlabel("Temps (x100s)")
plt.ylabel("Angle (rad)")
plt.legend()


plt.show()
"""