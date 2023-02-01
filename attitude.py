import numpy as np
import scipy.special as sp
import scipy.linalg as la

# Machine precision
EPS = np.finfo(float).eps
# Maximum iterations for A2
MAXITER = 10_000

def solve(I, w0, w, A0, t):
    if all(i == I[0] for i in I): # Spherically symmetric
        # Rotation
        w0_lab = A0.T @ w0 
        W0 = np.array([
            [0, w0_lab[2], - w0_lab[1]],
            [- w0_lab[2], 0, w0_lab[0]],
            [w0_lab[1], - w0_lab[0], 0]
        ])
        W = np.einsum('k,ij->kij', t, W0)
        P = la.expm(W) # nx3x3
        return np.einsum('ijk,kl->jli', P, A0) # 3x3xn
    elif all(i != I[0] for i in I[1:]): # Asymmetric
        # Conserved quantities
        l = np.sum([(I[i] * w0[i]) ** 2 for i in range(3)]) # L^2
        e = np.sum([I[i] * w0[i] ** 2 for i in range(3)]) # 2T
        L = np.sqrt(l)
        # Set intermediate axis as axis corresponding to median moment of inertia
        j = np.argsort(I)[len(I) // 2]
        k = (j + 1) % 3
        i = (j + 2) % 3
        I1, I2, I3 = I[i], I[j], I[k]
        w10, w20, w30 = w0[i], w0[j], w0[k]
        Lperp0 = np.sqrt((I1 * w10) ** 2 + (I2 * w20) ** 2)
        T1t0 = np.array([
            [I1 * I3 * w10 * w30 / (L * Lperp0), I2 * I3 * w20 * w30 / (L * Lperp0), - Lperp0 / L],
            [- I2 * w20 / Lperp0, I1 * w10 / Lperp0, 0],
            [I1 * w10 / L, I2 * w20 / L, I3 * w30 / L]
        ])
        # More useful quantities
        B = T1t0 @ A0
        le1 = l - e * I1
        le3 = l - e * I3
        s_w10 = np.sign(w10) if w10 != 0 else 1
        w2m = - s_w10 * np.sqrt(le3 / (I2 * (I2 - I3)))
        w3m = np.sign(w30) * np.sqrt(le1 / (I3 * (I3 - I1)))
        wp = np.sign(I2 - I3) * np.sign(w30) * np.sqrt(le1 * (I3 - I2) / np.prod(I))
        m = le3 * (I1 - I2) / (le1 * (I3 - I2))
        epsilon = sp.ellipkinc(np.arcsin(w20 / w2m), m)
        K = sp.ellipk(m)
        Kp = sp.ellipk(1 - m)
        q = np.exp(- np.pi * Kp / K)
        eta = np.sign(w30) * Kp - sp.ellipkinc(np.arcsin(I3 * w3m / L), 1 - m)
        xi = np.exp(np.pi * eta / K)
        # Compute A1, A2 parameters
        A2 = L / I1 + np.pi * wp * (xi + 1) / (2 * K * (xi - 1))
        n = 1
        dA2 = - np.pi * wp * q ** (2 * n) * (xi ** n - xi ** - n) / (K * (1 - q ** (2 * n)))
        A2 += dA2
        while np.abs(dA2) > EPS and n < MAXITER:
            n += 1
            dA2 = - np.pi * wp * q ** (2 * n) * (xi ** n - xi ** - n) / (K * (1 - q ** (2 * n)))
            A2 += dA2
        NT = int(np.log(EPS) / np.log(q) + 1.5)
        r0, i0 = 0, 0
        cr, ci = [], []
        for n in range(NT):
            Q = 2 * q ** (n * (n + 1) + .25)
            P = (2 * n + 1) * np.pi * eta / (2 * K)
            E = (2 * n + 1) * np.pi * epsilon / (2 * K)
            cr.append((- 1) ** n * Q * np.cosh(P))
            ci.append((- 1) ** (n + 1) * Q * np.sinh(P))
            r0 += cr[n] * np.sin(E)
            i0 += ci[n] * np.cos(E)
            if np.abs(cr[n] < EPS) and np.abs(ci[n] < EPS): # If converged earlier than expected
                NT = n + 1
                break
        A1 = np.arctan2(i0, r0)
        # Evolution in time
        Re_theta, Im_theta = 0, 0
        for n in range(NT):
            U = (2 * n + 1) * np.pi * (wp * t + epsilon) / (2 * K)
            Re_theta += cr[n] * np.sin(U)
            Im_theta += ci[n] * np.cos(U)
        C = np.cos(A1 + A2 * t)
        S = np.sin(A1 + A2 * t)
        theta = np.linalg.norm([Re_theta, Im_theta])
        cos_psi = (C * Re_theta + S * Im_theta) / theta
        sin_psi = (S * Re_theta - C * Im_theta) / theta
        w1, w2, w3 = w[i], w[j], w[k]
        Lperp = np.sqrt((I1 * w1) ** 2 + (I2 * w2) ** 2)
        T1 = np.array([
            [I1 * I3 * w1 * w3 / (L * Lperp), - I2 * w2 / Lperp, I1 * w1 / L],
            [I2 * I3 * w2 * w3 / (L * Lperp), I1 * w1 / Lperp, I2 * w2 / L],
            [- Lperp / L, np.zeros(len(t)), I3 * w3 / L]
        ]) # 3x3xn
        T2 = np.array([
            [cos_psi, sin_psi, np.zeros(len(t))],
            [- sin_psi, cos_psi, np.zeros(len(t))],
            [np.zeros(len(t)), np.zeros(len(t)), np.ones(len(t))]
        ]) # 3x3xn
        return np.einsum('ijk,jlk,ln->ink', T1, T2, B)
    else: # Axially symmetric
        # Set 3rd axis as axis corresponding to different moment of inertia
        k = np.argmax(np.abs(I - np.median(I)))
        i = (k + 1) % 3
        j = (k + 2) % 3
        # Rotation
        O = (I[k] / I[i] - 1) * w0[k]
        c, s = np.cos(O * t), np.sin(O * t)
        T1 = np.array([
            [c, - s, np.zeros(len(t))],
            [s, c, np.zeros(len(t))],
            [np.zeros(len(t)), np.zeros(len(t)), np.ones(len(t))]
        ]) # 3x3xn
        # Precession
        Op = np.array([I[i] * w0[i] for i in range(3)]) / I[i]
        Op0x = np.array([
            [0, Op[k], - Op[j]],
            [- Op[k], 0, Op[i]],
            [Op[j], - Op[i], 0]
        ])
        Opx = np.einsum('k,ij->kij', t, Op0x)
        T2 = la.expm(Opx) # nx3x3
        return np.einsum('ijk,kjl,lm->imk', T1, T2, A0) # 3x3xn