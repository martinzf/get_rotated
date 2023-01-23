import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Duration
t = np.arange(
    0, 
    request_float('Input simulation duration (s): ', False),
    .02)

if all(i != I[0] for i in I[1:]): # Assymetric
    # Correct ordering of basis
    j = np.argsort(I)[len(I) // 2]
    k = (j + 1) % 3
    i = (j + 2) % 3
    # Conserved quantities
    l = np.sum([(I[i] * w0[i]) ** 2 for i in range(3)]) # L^2
    e = np.sum([I[i] * w0[i] ** 2 for i in range(3)]) # 2T
    # Change of variables
    tau = t * np.sqrt( (I[i] * e - l) * (I[j] - I[k]) / (np.prod(I)))
    # Initial conditions
    s0 = w0[j] * np.sqrt(I[j] * (I[j] - I[k]) / (l - e * I[k]))
    m = (I[i] - I[j]) * (l - I[k] * e) / ((I[j] - I[k]) * (I[i] * e - l))
    tau += sp.ellipkinc(np.arcsin(s0), m)
    # Solution
    w = np.empty((3, len(t)))
    sn, cn, dn, _ = sp.ellipj(tau, m)
    w[i] = np.sqrt((l - I[k] * e) / (I[i] * (I[i] - I[k]))) * cn  
    w[j] = np.sqrt((l - I[k] * e) / (I[j] * (I[j] - I[k]))) * sn  
    w[k] = np.sqrt((I[i] * e - l) / (I[k] * (I[i] - I[k]))) * dn 
elif all(i == I[0] for i in I): # Spherically symmetric
    w0 = np.reshape(w0, (3, 1))
    w = np.tile(w0, len(t))
else: # Axially symmetric
    # Correct ordering of basis
    k = np.argmax(np.abs(I-np.median(I))) 
    i = (k + 1) % 3
    j = (k + 2) % 3
    w = np.empty((3, len(t)))
    # Precession of w around symmetry axis (in body frame)
    O = (I[k] - I[i]) / I[i] 
    w[i] = np.cos(O * t) * w0[i] - np.sin(O * t) * w0[j]
    w[j] = np.sin(O * t) * w0[i] + np.cos(O * t) * w0[j]
    w[k] = w0[k] 
# Angular momentum
L = np.reshape(I, (3, 1)) * w

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2, 1, 1, projection='3d')
ax2 = fig.add_subplot(2, 1, 2)

# Axis one
ax1.set_title('Body frame', fontdict={'size': 15})
lim1 = np.max([w, L])
ax1.set_xlim([-lim1, lim1])
ax1.set_ylim([-lim1, lim1])
ax1.set_zlim([-lim1, lim1])
# Hide everything
ax1.xaxis.set_pane_color((1, 1, 1, 0))
ax1.yaxis.set_pane_color((1, 1, 1, 0))
ax1.zaxis.set_pane_color((1, 1, 1, 0))
ax1._axis3don = False

# Axis two
ax2.set_xlim([0, t[-1]])
ax2.set_ylim([np.min(w), np.max(w)])

# Animation
ln1, = ax1.plot([], [], [], color='b', lw=1.2, label=r'$\vec{\omega}$')
ln2, = ax1.plot([], [], [], color='r', lw=1.2, label=r'$\vec{L}$')
angular_velocity = ax1.quiver([], [], [], [], [], [], color='b')
angular_momentum = ax1.quiver([], [], [], [], [], [], color='r')
omega1, = ax2.plot([], [], label=r'$\omega_1$')
omega2, = ax2.plot([], [], label=r'$\omega_2$')
omega3, = ax2.plot([], [], label=r'$\omega_3$')
def func(i):
    global angular_velocity
    global angular_momentum
    angular_velocity.remove()
    angular_momentum.remove()
    angular_velocity = ax1.quiver(0, 0, 0, *w[:, i], color='b', arrow_length_ratio=.1, alpha=.6)
    angular_momentum = ax1.quiver(0, 0, 0, *L[:, i], color='r', arrow_length_ratio=.1, alpha=.6)
    ln1.set_data_3d(w[:, :i])
    ln2.set_data_3d(L[:, :i])
    omega1.set_data(t[:i], w[0, :i])
    omega2.set_data(t[:i], w[1, :i])
    omega3.set_data(t[:i], w[2, :i])

def init_func():
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax2.set_xlabel('t (s)')
    ax2.set_ylabel('$\omega$ (rad/s)')
    # Style axis one
    val = [lim1 * 1.5, 0, 0]
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        xyz = [-val[i-0], -val[i-1], -val[i-2]]
        uvw = [2 * val[i-0], 2 * val[i-1], 2 * val[i-2]]
        ax1.quiver(*xyz, *uvw, color='k', arrow_length_ratio=.05)
        ax1.text(val[i-0], val[i-1], val[i-2], labels[i], fontsize=15)

ani = FuncAnimation(fig, func, frames=len(t), init_func=init_func, interval=50)
ani.save('rb_rotation.gif', writer='pillow', fps=50, dpi=100)

plt.show()