import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Time step
dt = .02

def request_float(prompt: str, positive: bool) -> float:
    # Gets user input
    while True:
        try:
            answer = float(input(prompt))
            if not(positive):
                return answer
            if answer > 0:
                return answer
            print('Input must be strictly positive.')
        except ValueError:
            print(f'Input must be a float.') 

print(
    'We have taken a body frame of principal axes of inertia, '
    'centered at a point fixed in the body frame and an external inertial frame.'
    )

# Moment of inertia tensor in principal axes of inertia body frame
print('Input the inertia tensor in the body frame.')
I = []
for i in range(1, 4):
    I.append(request_float(f'Moment of inertia I{i} (kg m^2): ', True))

# Initial angular velocity
print('Input the initial angular velocity in the body frame.')
w0 = []
for i in range(1, 4):
    w0.append(request_float(f'w{i} (rad/s): ', False))

if (w0.count(0) == 2) or all(i == I[0] for i in I): # w along principal axis or spherical symmetry
    t = np.arange(0, 3, dt)
    w0 = np.reshape(w0, (3, 1))
    w = np.tile(w0, len(t))
elif all(i != I[0] for i in I[1:]): # Assymetric I
    # Correct ordering of basis
    j = np.argsort(I)[len(I) // 2]
    k = (j + 1) % 3
    i = (j + 2) % 3
    # Conserved quantities
    l = np.sum([(I[i] * w0[i]) ** 2 for i in range(3)]) # L^2
    e = np.sum([I[i] * w0[i] ** 2 for i in range(3)]) # 2T
    # Change of variables
    m = (I[i] - I[j]) * (l - I[k] * e) / ((I[j] - I[k]) * (I[i] * e - l))
    K = 4 * sp.ellipk(m)
    conversion = np.sqrt( (I[i] * e - l) * (I[j] - I[k]) / (np.prod(I)))
    t = np.arange(0, K / conversion, dt)
    tau = t * conversion
    # Initial conditions
    s0 = w0[j] * np.sqrt(I[j] * (I[j] - I[k]) / (l - e * I[k]))
    tau += sp.ellipkinc(np.arcsin(s0), m)
    # Solution
    w = np.empty((3, len(t)))
    sn, cn, dn, _ = sp.ellipj(tau, m)
    w[i] = np.sqrt((l - I[k] * e) / (I[i] * (I[i] - I[k]))) * cn  
    w[j] = np.sqrt((l - I[k] * e) / (I[j] * (I[j] - I[k]))) * sn  
    w[k] = np.sqrt((I[i] * e - l) / (I[k] * (I[i] - I[k]))) * dn 
else: # Axially symmetric I
    # Correct ordering of basis
    k = np.argmax(np.abs(I-np.median(I))) 
    i = (k + 1) % 3
    j = (k + 2) % 3
    # Precession of w around symmetry axis (in body frame)
    O = (I[k] - I[i]) / I[i] 
    t = np.arange(0, 2 * np.pi / O, dt)
    w = np.empty((3, len(t)))
    w[i] = np.cos(O * t) * w0[i] - np.sin(O * t) * w0[j]
    w[j] = np.sin(O * t) * w0[i] + np.cos(O * t) * w0[j]
    w[k] = w0[k] 
# Angular momentum (invariant direction)
L = np.reshape(I, (3, 1)) * w
# Spherical coords for L
theta = np.arccos(L[2] / np.sum(L ** 2, axis=0))
phi = np.arctan2(L[1], L[0])
# Rotation matrices, L -> z axis
cost = np.cos(theta)
sint = np.sin(theta)
cosp = np.cos(phi)
sinp = np.sin(phi)
R = np.array([
    [cosp * cost, sinp, -cosp * sint],
    [-sinp * cost, cosp, sinp * cost],
    [sint, np.zeros(len(t)), cost]
])
# Inertial frame coordinates for w
w_inertial = np.einsum('ijk,jk->jk', R, w)

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax4 = fig.add_subplot(2, 2, 4)

# Axes 1 and 3
ax1.set_title('Body frame', fontdict={'size': 15})
lim1 = np.max(np.abs([w, L]))
ax1.set_xlim([-lim1, lim1])
ax1.set_ylim([-lim1, lim1])
ax1.set_zlim([-lim1, lim1])
ax3.set_title('Inertial frame', fontdict={'size': 15})
lim3 = np.max(np.abs(w_inertial))
ax3.set_xlim([-lim3, lim3])
ax3.set_ylim([-lim3, lim3])
ax3.set_zlim([-lim3, lim3])
# Style axes
val1 = [lim1 * 1.5, 0, 0]
val3 = [lim3 * 1.5, 0, 0]
labels = ['X', 'Y', 'Z']
for i in range(3):
    xyz = [-val1[i-0], -val1[i-1], -val1[i-2]]
    uvw = [2 * val1[i-0], 2 * val1[i-1], 2 * val1[i-2]]
    ax1.quiver(*xyz, *uvw, color='k', arrow_length_ratio=.05)
    ax1.text(val1[i-0], val1[i-1], val1[i-2], labels[i], fontsize=15)
    ax3.quiver(*xyz, *uvw, color='k', arrow_length_ratio=.05)
    ax3.text(val3[i-0], val3[i-1], val3[i-2], labels[i], fontsize=15)
# Hide everything default
ax1.xaxis.set_pane_color((1, 1, 1, 0))
ax1.yaxis.set_pane_color((1, 1, 1, 0))
ax1.zaxis.set_pane_color((1, 1, 1, 0))
ax1._axis3don = False
ax3.xaxis.set_pane_color((1, 1, 1, 0))
ax3.yaxis.set_pane_color((1, 1, 1, 0))
ax3.zaxis.set_pane_color((1, 1, 1, 0))
ax3._axis3don = False

# Axes 2 and 4
ax2.set_xlabel('t (s)')
ax2.set_ylabel('$\omega$ (rad/s)')
minw = np.min(w)
maxw = np.max(w)
ax2.set_xlim([0, t[-1]])
ax2.set_ylim([minw - .1 * np.abs(minw), maxw + .1 * np.abs(maxw)])
ax4.set_xlabel('t (s)')
ax4.set_ylabel('$\omega$ (rad/s)')
minw_i = np.min(w_inertial)
maxw_i = np.max(w_inertial)
ax4.set_xlim([0, t[-1]])
ax4.set_ylim([minw_i - .1 * np.abs(minw_i), maxw_i + .1 * np.abs(maxw_i)])

# Animation
angular_velocity, = ax1.plot([], [], [], color='b', lw=2.5, alpha=.6)
angular_momentum, = ax1.plot([], [], [], color='r', lw=2.5, alpha=.6)
ln_w, = ax1.plot([], [], [], color='b', lw=1.2, label=r'$\vec{\omega}$')
ln_L, = ax1.plot([], [], [], color='r', lw=1.2, label=r'$\vec{L}$')
omega1, = ax2.plot([], [], label=r'$\omega_1$')
omega2, = ax2.plot([], [], label=r'$\omega_2$')
omega3, = ax2.plot([], [], label=r'$\omega_3$')
angular_velocity_i, = ax3.plot([], [], [], color='b', lw=2.5, alpha=.6)
ln_w_i, = ax3.plot([], [], [], color='b', lw=1.2, label=r'$\vec{\omega}$')
omega1_i, = ax4.plot([], [], label=r'$\omega_1$')
omega2_i, = ax4.plot([], [], label=r'$\omega_2$')
omega3_i, = ax4.plot([], [], label=r'$\omega_3$')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')
def func(i):
    angular_velocity.set_data_3d(*zip(np.zeros(3), w[:, i]))
    angular_momentum.set_data_3d(*zip(np.zeros(3), L[:, i]))
    ln_w.set_data_3d(w[:, :i])
    ln_L.set_data_3d(L[:, :i])
    omega1.set_data(t[:i], w[0, :i])
    omega2.set_data(t[:i], w[1, :i])
    omega3.set_data(t[:i], w[2, :i])
    angular_velocity_i.set_data_3d(*zip(np.zeros(3), w_inertial[:, i]))
    ln_w_i.set_data_3d(w_inertial[:, :i])
    omega1_i.set_data(t[:i], w_inertial[0, :i])
    omega2_i.set_data(t[:i], w_inertial[1, :i])
    omega3_i.set_data(t[:i], w_inertial[2, :i])
    return \
        angular_velocity, \
        angular_momentum, \
        ln_w, \
        ln_L, \
        omega1, \
        omega2, \
        omega3, \
        angular_velocity_i, \
        ln_w_i, \
        omega1_i, \
        omega2_i, \
        omega3_i

ani = FuncAnimation(fig, func, frames=len(t), interval=50, blit=True)
ani.save('rb_rotation.gif', writer='pillow', fps=50, dpi=100)

plt.show()