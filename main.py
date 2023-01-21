import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

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
    # Finding median value
    j = np.argsort(I)[len(I) // 2]
    k = (j + 1) % 3
    i = (j + 2) % 3
    # Conserved quantities
    l = np.sum([(I[i] * w0[i]) ** 2 for i in range(3)]) # L^2
    e = np.sum([I[i] * w0[i] ** 2 for i in range(3)]) # 2E
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
    # Precession of w around symmetry axis (in body frame)
    w = np.empty((3, len(t)))
    O = (I[k] - I[i]) / I[i] 
    w[i] = np.cos(O * t) * w0[i] - np.sin(O * t) * w0[j]
    w[j] = np.sin(O * t) * w0[i] + np.cos(O * t) * w0[j]
    w[k] = w0[k] * np.ones(len(t))

plt.plot(t, w[0])
plt.plot(t, w[1])
plt.plot(t, w[2])
plt.legend(['w1', 'w2', 'w3'])
plt.show()