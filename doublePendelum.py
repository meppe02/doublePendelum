import numpy as np
import sympy as smp
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation

"""RÖRELSE EKVATIONERNA"""

# Symboler för variabler och parametrar
t, g = smp.symbols("t g")
m1, m2 = smp.symbols("m1 m2")
L1, L2 = smp.symbols("L1, L2")

# Symboler för vinklar och deras derivator
the1, the2 = smp.symbols(r"\theta_1, \theta_2", cls=smp.Function)
the1 = the1(t)
the2 = the2(t)

# Derivator av vinklar
the1_d = smp.diff(the1, t)
the2_d = smp.diff(the2, t)
the1_dd = smp.diff(the1_d, t)
the2_dd = smp.diff(the2_d, t)

# Beräkning av x- och y-koordinater för pendelstavar
x1 = L1 * smp.sin(the1)
y1 = -L1 * smp.cos(the1)
x2 = L1 * smp.sin(the1) + L2 * smp.sin(the2)
y2 = -L1 * smp.cos(the1) - L2 * smp.cos(the2)

# RÖRELSE ENERGIN
T1 = 1 / 2 * m1 * (smp.diff(x1, t) ** 2 + smp.diff(y1, t) ** 2)
T2 = 1 / 2 * m2 * (smp.diff(x2, t) ** 2 + smp.diff(y2, t) ** 2)
T = T1 + T2

# POTENTIAL ENERGI
V1 = m1 * g * y1
V2 = m2 * g * y2
V = V1 + V2

# Damping coefficients
C1 = 0.02  # Damping coefficient for pendulum 1
C2 = 0.04  # Damping coefficient for pendulum 2

# Damping forces in the x-direction
F1_x = C1 * smp.diff(x1, t)
F2_x = C2 * smp.diff(x2, t)
# Lagrangian without damping terms
L = T - V

# Lagrangian with damping terms
LE1 = smp.diff(smp.diff(L, the1_d), t).simplify() - smp.diff(L, the1) + F1_x
LE2 = smp.diff(smp.diff(L, the2_d), t).simplify() - smp.diff(L, the2) + F2_x

# Lösning av differentialekvationerna med sympy
sols = smp.solve([LE1, LE2], (the1_dd, the2_dd), simplify=False, rational=False)

# Skapar funktioner av differentialekvationerna
dz1dt_f = smp.lambdify(
    (t, g, m1, m2, L1, L2, the1, the2, the1_d, the2_d), sols[the1_dd]
)
dz2dt_f = smp.lambdify(
    (t, g, m1, m2, L1, L2, the1, the2, the1_d, the2_d), sols[the2_dd]
)
dthe1dt_f = smp.lambdify(the1_d, the1_d)
dthe2dt_f = smp.lambdify(the2_d, the2_d)


# Funktion för att lösa differentialekvationerna
def dSdt(S, t, g, m1, m2, L1, L2):
    the1, z1, the2, z2 = S
    return [
        dthe1dt_f(z1),
        dz1dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
        dthe2dt_f(z2),
        dz2dt_f(t, g, m1, m2, L1, L2, the1, the2, z1, z2),
    ]


"""GRAFFRAMSTÄLLNING"""

frames = 10000
time = 200
t = np.linspace(0, time, frames)

g = 9.82
m1 = 0.2
m2 = 0.2
L1 = 0.3
L2 = 0.3

# Löser differentialekvationerna med odeint
thetaIn_1 = 160  # Degrees
thetaIn_2 = 130 # Degrees

ans = odeint(
    dSdt,
    y0=[np.radians(thetaIn_1), 0, np.radians(thetaIn_2), 0],
    t=t,
    args=(g, m1, m2, L1, L2),
)

the1 = ans.T[0]
the2 = ans.T[2]

thetaIn_1_ = thetaIn_1 + 0.1  # Degrees
thetaIn_2_ = thetaIn_1_  # Degrees

ans_ = odeint(
    dSdt,
    y0=[np.radians(thetaIn_1_), 0, np.radians(thetaIn_2_), 0],
    t=t,
    args=(g, m1, m2, L1, L2),
)

the1_ = ans_.T[0]
the2_ = ans_.T[2]

r1 = -1 * L1 * np.array([np.sin(the1), np.cos(the1)])
r2 = r1 + -1 * L2 * np.array([np.sin(the2), np.cos(the2)])

fig, axs = plt.subplots(1, 2, figsize=(10, 7))

# Inställningar för den första plotten
axs[0].set_xlim([-L1 - L2, L1 + L2])
axs[0].set_ylim([-L1 - L2, L1 + L2])
axs[0].set_aspect("equal")
axs[0].grid()
axs[0].set_title("Double Pendelum Animation")
axs[0].set_xlabel("x-pos i meter")
axs[0].set_ylabel("y-pos i meter")
axs[0].tick_params(axis="both", which="major")
(pointT,) = axs[0].plot([], [], "o", lw=2)
(pointB,) = axs[0].plot([], [], "o", lw=2)
(line1,) = axs[0].plot([], [], lw=2)
(line2,) = axs[0].plot([], [], lw=2)
(trace,) = axs[0].plot([], [], lw=1, color="gray")


xtrace = []
ytrace = []

# Inställningar för den andra plotten
maxngle = np.max([the1, the2])
axs[1].set_xlim(-maxngle, maxngle)
axs[1].set_ylim([-maxngle, maxngle])
axs[1].set_aspect("equal")
axs[1].grid()
axs[1].set_title("Generalised Cordinates")
axs[1].set_xlabel("Vinkel för pendel 1")
axs[1].set_ylabel("Vinkel för pendel 2")
axs[1].tick_params(axis="both", which="major")
(generalised,) = axs[1].plot([], [], "-b", markersize=2)

g_the1 = []
g_the2 = []


def animate(i):
    line1.set_data([0, r1[0, i]], [0, r1[1, i]])
    line2.set_data([r1[0, i], r2[0, i]], [r1[1, i], r2[1, i]])

    # Uppdatera positionen för pointT och pointB
    pointT.set_data([r1[0, i]], [r1[1, i]])
    pointB.set_data([r2[0, i]], [r2[1, i]])

    xtrace.append(r2[0, i])
    ytrace.append(r2[1, i])
    trace.set_data(xtrace, ytrace)

    while the1[i] > np.pi:
        the1[i] -= 2 * np.pi
    while the1[i] < -np.pi:
        the1[i] += 2 * np.pi
    while the2[i] > np.pi:
        the2[i] -= 2 * np.pi
    while the2[i] < -np.pi:
        the2[i] += 2 * np.pi

    if i > 0 and (
        np.abs(the1[i] - the1[i - 1]) > np.pi or np.abs(the2[i] - the2[i - 1]) > np.pi
    ):
        g_the1.append(None)
        g_the2.append(None)
    else:
        g_the1.append(the1[i])
        g_the2.append(the2[i])

    generalised.set_data(g_the1, g_the2)

    return line1, line2, trace, generalised, pointT


ani = FuncAnimation(fig, animate, frames=frames, interval=20)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
plt.tick_params(axis="both")

ax.plot(t, np.degrees(the1), label=r"$\theta_1$")
ax.plot(t, np.degrees(the2), label=r"$\theta_2$")
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")

ax.legend(loc="lower right", fontsize=30)
ax.tick_params(axis="both", which="major")


ax.set_title(
    "Graf för "
    + r"$\theta_1=$"
    + f"{thetaIn_1}"
    + r"$\degree$ och $\theta_2=$ $\sqrt{2}\cdot$"
    + f"{thetaIn_1}"
    + r"$\degree$",
)
"""
# Plotting the third set of data and providing labels
ax[1].plot(t[:len(t)//5], (the1 + the2)[:len(t)//5], label=r"$\theta_1 + \theta_2$")

# Adding legend for the second subplot and positioning it to the right in the upper corner
ax[1].legend(loc="lower right", fontsize=14)

# Adding title for the second subplot
ax[1].set_title(r"Superpositionsprincipen för $\theta_1$ och $\theta_2 (\theta_1+\theta_2)$", fontsize=16)
"""
# Set x-axis labels for both subplots

ax.set_xlabel("Tid [s]")
ax.set_ylabel("Vinklar [grader]")
plt.tight_layout()
plt.grid()
plt.show()

from scipy.fft import fft, fftfreq


fft_the1 = fft(the1[len(the1) // 2 :])
fft_the2 = fft(the2[len(the2) // 2 :])
dt = t[1] - t[0]


freq = fftfreq(len(t) // 2, d=dt)
x_axis_range = (-3, 3)  
mask = (freq >= x_axis_range[0]) & (freq <= x_axis_range[1])
freq = freq[mask]


fig = plt.figure(figsize=(10, 7)) 


plt.plot(freq, abs(fft_the1[mask]) / len(t), label=r"$\theta_1$")
plt.plot(freq, abs(fft_the2[mask]) / len(t), label=r"$\theta_2$")
plt.tick_params(axis="both", which="major")
plt.title(
    "FFT av "
    r"$\theta_1=$"
    + f"{thetaIn_1}"
    + r"$\degree$ och "
    + r"$\sqrt{2}\cdot$"
    + f"{thetaIn_1}"
    + r"$\degree$",

)
plt.xlabel("Frekvens (Hz)")
plt.ylabel("Amplitud")
plt.xlim(x_axis_range)
plt.grid(True)
plt.legend(fontsize=30)

# Show the plot
plt.show()
