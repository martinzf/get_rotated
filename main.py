import numpy as np
import rb_rotation
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('fast')

# Time step
DT = .033 

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

def get_data() -> tuple[np.array]:
    print(
        'We shall employ two reference frames: \n'
        '1. A body frame of principal axes of inertia, '
        'centered at a point fixed in the body frame and an external inertial frame. \n'
        '2. The inertial lab frame.'
        )
    # Moment of inertia tensor in body frame
    print('Input the (diagonal) inertia tensor in the BODY FRAME.')
    I = []
    for i in range(1, 4):
        I.append(request_float(f'Moment of inertia I{i} (kg m^2): ', True))
    I = np.array(I)
    # Initial angular velocity
    print('Input the initial angular velocity in the LAB FRAME.')
    w0_lab = []
    for i in range(1, 4):
        w0_lab.append(request_float(f'w{i} (rad/s): ', False))
    w0_lab = np.array(w0_lab)
    # Initial transposed (inverse) attitude matrix
    print('Input the initial orientation of the body in the LAB FRAME: Rz(yaw) Ry(pitch) Rx(roll). ')
    roll = request_float('Roll (rad): ', False)
    pitch = request_float('Pitch (rad): ', False)
    yaw = request_float('Yaw (rad): ', False)
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    A0 = np.array([
        [cz * cy, sz * cy, - sy],
        [cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx],
        [cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx]
    ])
    w0_body = A0 @ w0_lab
    # Duration
    t = np.arange(0, request_float("Input the simulation's duration t (s): ", True), DT)
    return I, w0_body, A0, t

# Plotting & animating
def init_figure(
    t: np.array, w_body: np.array, w_lab: np.array
    ) -> tuple[mpl.figure.Figure, float, mpl.lines.Line2D]:
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4)
    # Axes 1 and 3
    ax1.set_title('Body frame', fontdict={'size': 15})
    lim1 = np.max(np.abs(w_body))
    ax1.set_xlim(-lim1, lim1)
    ax1.set_ylim(-lim1, lim1)
    ax1.set_zlim(-lim1, lim1)
    ax3.set_title('Lab frame', fontdict={'size': 15})
    lim3 = np.max(np.abs(w_lab))
    ax3.set_xlim(-lim3, lim3)
    ax3.set_ylim(-lim3, lim3)
    ax3.set_zlim(-lim3, lim3)
    # Style axes
    val1 = [lim1 * 1.5, 0, 0]
    val3 = [lim3 * 1.5, 0, 0]
    labels1 = ["$X_1$", "$X_2$", "$X_3$"]
    labels3 = ["$X_1'$", "$X_2'$", "$X_3'$"]
    for i in range(3):
        xyz = [-val1[i-0], -val1[i-1], -val1[i-2]]
        uvw = [2 * val1[i-0], 2 * val1[i-1], 2 * val1[i-2]]
        ax1.quiver(*xyz, *uvw, color='k', arrow_length_ratio=.05)
        ax1.text(val1[i-0], val1[i-1], val1[i-2], labels1[i], fontsize=15)
        xyz = [-val3[i-0], -val3[i-1], -val3[i-2]]
        uvw = [2 * val3[i-0], 2 * val3[i-1], 2 * val3[i-2]]
        ax3.quiver(*xyz, *uvw, color='k', arrow_length_ratio=.05)
        ax3.text(val3[i-0], val3[i-1], val3[i-2], labels3[i], fontsize=15)
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
    minw_body = np.min(w_body)
    minw_body = minw_body if np.abs(minw_body) > .1 else -.1
    maxw_body = np.max(w_body)
    maxw_body = maxw_body if np.abs(maxw_body) > .1 else .1
    ax2.set_xlim([0, t[-1]])
    ax2.set_ylim([minw_body - .1 * np.abs(minw_body), maxw_body + .1 * np.abs(maxw_body)])
    ax4.set_xlabel('t (s)')
    ax4.set_ylabel('$\omega$ (rad/s)')
    minw_lab = np.min(w_lab)
    minw_lab = minw_lab if np.abs(minw_lab) > .1 else -.1
    maxw_lab = np.max(w_lab)
    maxw_lab = maxw_lab if np.abs(maxw_lab) > .1 else .1
    ax4.set_xlim([0, t[-1]])
    ax4.set_ylim([minw_lab - .1 * np.abs(minw_lab), maxw_lab + .1 * np.abs(maxw_lab)])
    # Artists
    axis_length = .7 * np.linalg.norm(w_body[:, 0])
    angular_velocity_b, = ax1.plot([], [], [], color='b', lw=2.5, alpha=.6, label=r'$\vec{\omega}$')
    line_w_b, = ax1.plot([], [], [], color='b', lw=1.2)
    omega1_b, = ax2.plot([], [], label=r'$\omega_1$')
    omega2_b, = ax2.plot([], [], label=r'$\omega_2$')
    omega3_b, = ax2.plot([], [], label=r'$\omega_3$')
    angular_velocity_l, = ax3.plot([], [], [], color='b', lw=2.5, alpha=.6, label=r"$\vec{\omega}$")
    e1, = ax3.plot([], [], [], lw=2.5, alpha=.6, label=r'$\vec{e}_1$')
    e2, = ax3.plot([], [], [], lw=2.5, alpha=.6, label=r'$\vec{e}_2$')
    e3, = ax3.plot([], [], [], lw=2.5, alpha=.6, label=r'$\vec{e}_3$') 
    line_w_l, = ax3.plot([], [], [], color='b', lw=1.2)
    omega1_l, = ax4.plot([], [], label=r"$\omega_1$")
    omega2_l, = ax4.plot([], [], label=r"$\omega_2$")
    omega3_l, = ax4.plot([], [], label=r"$\omega_3$")
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    ax4.legend(loc='upper right')
    return \
        fig, \
        axis_length, \
        angular_velocity_b, \
        line_w_b, \
        omega1_b, \
        omega2_b, \
        omega3_b, \
        angular_velocity_l, \
        e1, e2, e3, \
        line_w_l, \
        omega1_l, \
        omega2_l, \
        omega3_l

def animate(i: int) -> tuple[mpl.lines.Line2D]:
    angular_velocity_b.set_data_3d(*zip(np.zeros(3), w_body[:, i]))
    line_w_b.set_data_3d(w_body[:, :i])
    omega1_b.set_data(t[:i], w_body[0, :i])
    omega2_b.set_data(t[:i], w_body[1, :i])
    omega3_b.set_data(t[:i], w_body[2, :i])
    angular_velocity_l.set_data_3d(*zip(np.zeros(3), w_lab[:, i]))
    e1.set_data_3d(*zip(np.zeros(3), axis_length * A[0, :, i]))
    e2.set_data_3d(*zip(np.zeros(3), axis_length * A[1, :, i]))
    e3.set_data_3d(*zip(np.zeros(3), axis_length * A[2, :, i]))
    line_w_l.set_data_3d(w_lab[:, :i])
    omega1_l.set_data(t[:i], w_lab[0, :i])
    omega2_l.set_data(t[:i], w_lab[1, :i])
    omega3_l.set_data(t[:i], w_lab[2, :i])
    return \
        angular_velocity_b, \
        line_w_b, \
        omega1_b, \
        omega2_b, \
        omega3_b, \
        angular_velocity_l, \
        e1, e2, e3, \
        line_w_l, \
        omega1_l, \
        omega2_l, \
        omega3_l

if __name__ == '__main__':
    I, w0_body, A0, t = get_data()
    w_body, A = rb_rotation.solve(I, w0_body, A0, t)
    w_lab = np.einsum('jik,jk->ik', A, w_body) # A.T @ w_body
    fig, \
    axis_length, \
    angular_velocity_b, \
    line_w_b, \
    omega1_b, \
    omega2_b, \
    omega3_b, \
    angular_velocity_l, \
    e1, e2, e3, \
    line_w_l, \
    omega1_l, \
    omega2_l, \
    omega3_l \
    = init_figure(t, w_body, w_lab)
    ani = FuncAnimation(fig, animate, frames=len(t), interval=DT*1e3, blit=True)
    plt.show()