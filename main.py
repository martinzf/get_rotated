import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def request(type: callable, prompt: str, positive: bool) -> float:
    # Gets user input
    while True:
        try:
            answer = type(input(prompt))
            if not(positive):
                return answer
            if answer > 0:
                return answer
            print('Input must be strictly positive.')
        except ValueError:
            print(f'Input must be {type}.') 

print(
    'We have taken a body frame of principal axes of inertia, '
    'centered at a point fixed in the body frame and an external inertial frame.'
    )

# Moment of inertia tensor in principal axes of inertia body frame
print('Input the inertia tensor in the body frame.')
I = []
for i in range(1, 4):
    I.append(request(float, f'Moment of inertia I{i} (kg m^2): ', True))

# Initial angular velocity
print('Input the initial angular velocity in the body frame.')
w0 = []
for i in range(1, 4):
    w0.append(request(float, f'w{i} (rad/s): ', False))

# Duration
t = np.arange(
    0, 
    request(float, 'Input simulation duration (s): ', False),
    .02)

tol = 1e-5 # Tolerance for checking symmetry
if all(np.abs(i - I[0]) > tol for i in I[1:]): # Assymetric
    # Correct ordering of basis
    j = np.argsort(I)[len(I) // 2]
    k = (j + 1) % 3
    i = (j + 2) % 3
    # Conserved quantities
    l = np.sum([(I[i] * w0[i]) ** 2 for i in range(3)]) # L^2
    e = np.sum([I[i] * w0[i] ** 2 for i in range(3)]) # 2T
    # Change of variables
    tau = t * np.sqrt( (I[i] * e - l) * (I[j] - I[k]) / (np.prod(I)))
    m = np.sqrt( (I[i] - I[j]) * (l - I[k] * e) / ((I[j] - I[k]) * (I[i] * e - l)))
    # Solution
    w = np.empty((3, len(t)))
    sn, cn, dn, _ = sp.ellipj(tau, m)
    w[i] = (l - I[k] * e) * cn / (I[i] * (I[i] - I[k]))  
    w[j] = (l - I[k] * e) * sn / (I[j] * (I[j] - I[k]))  
    w[k] = (I[i] * e - l) * dn / (I[k] * (I[i] - I[k])) 
elif all(np.abs(i - I[0]) <= tol for i in I): # Spherically symmetric
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

lim = np.max(w)
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
ax.set_title('Body frame', fontdict={'size': 20})
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([-lim, lim])
# Hide everything
ax.xaxis.set_pane_color((1, 1, 1, 0))
ax.yaxis.set_pane_color((1, 1, 1, 0))
ax.zaxis.set_pane_color((1, 1, 1, 0))
ax._axis3don = False
# Centered axes
val = [lim * 2, 0, 0]
labels = ['X', 'Y', 'Z']
for i in range(3):
    x = [val[i-0], -val[i-0]]
    y = [val[i-1], -val[i-1]]
    z = [val[i-2], -val[i-2]]
    ax.plot(x, y, z, 'k')
    ax.text(val[i-0], val[i-1], val[i-2], labels[i], fontsize=15)

ln, = ax.plot([], [], [], 'r', alpha=.5)
angular_velocity = ax.quiver([], [], [], [], [], [])
def animate(i):
    global angular_velocity
    angular_velocity.remove()
    angular_velocity = ax.quiver(0, 0, 0, *w[:, i])
    ln.set_data_3d(w[:, :i])

ani = FuncAnimation(fig, animate, frames=len(t), interval=50)
ani.save('rb_rotation.gif', writer='pillow', fps=50, dpi=100)