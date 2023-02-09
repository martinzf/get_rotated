import numpy as np
import scipy.special as sp
import scipy.linalg as la

# Machine precision
EPS = np.finfo(float).eps
# Maximum iterations for A2 coefficient
MAXITER = 10_000

def solve(I: np.array, w0: np.array, A0: np.array, t: np.array) -> tuple[np.array]:
    if (np.count_nonzero(w0 == 0) == 2) or \
    (len(set(I)) == 2 and w0[np.argmax(np.abs(I - np.median(I)))] == 0) or \
    (len(set(I)) == 1): # w along principal axis, axial symmetry and w3 = 0, or spherical symmetry
        # Rotation
        W0 = np.array([
            [0, w0[2], - w0[1]],
            [- w0[2], 0, w0[0]],
            [w0[1], - w0[0], 0]
        ])
        W = np.einsum('k,ij->kij', t, W0)
        P = la.expm(W) # nx3x3
        A = np.einsum('ijk,kl->jli', P, A0) # 3x3xn
        # Angular velocity
        w0 = np.reshape(w0, (3, 1))
        w = np.tile(w0, len(t))
        return w, A
    elif len(set(I)) == 2: # Axially symmetric
        # Set 3rd axis as axis corresponding to different moment of inertia
        k = np.argmax(np.abs(I - np.median(I)))
        i = (k + 1) % 3
        j = (k + 2) % 3
        # Precession of w around symmetry axis
        O = (I[k] / I[i] - 1) * w0[k]
        c, s = np.cos(O * t), np.sin(O * t)
        w = np.empty((3, len(t)))
        w[i] = c * w0[i] - s * w0[j]
        w[j] = s * w0[i] + c * w0[j]
        w[k] = w0[k] * np.ones(len(t))
        z, o = np.zeros(len(t)), np.ones(len(t))
        if k == 2:
            T1 = np.array([
                [c, - s, z],
                [s, c, z],
                [z, z, o]
            ]) # 3x3xn
        elif k == 1:
            T1 = np.array([
                [c, z, s],
                [z, o, z],
                [- s, z, c]
            ])
        else:
            T1 = np.array([
                [o, z, z],
                [z, c, - s],
                [z, s, c]
            ]) 
        # Precession of symmetry axis around L
        Op = np.array([I[i] * w0[i] for i in range(3)]) / I[i]
        Op0x = np.array([
            [0, Op[2], - Op[1]],
            [- Op[2], 0, Op[0]],
            [Op[1], - Op[0], 0]
        ])
        Opx = np.einsum('k,ij->kij', t, Op0x)
        T2 = la.expm(Opx) # nx3x3
        A = np.einsum('ijk,kjl,lm->imk', T1, T2, A0) # 3x3xn
        return w, A
    else: # Asymmetric
        # Set intermediate axis as axis corresponding to median moment of inertia
        j = np.argsort(I)[1]
        k = (j + 1) % 3
        i = (j + 2) % 3
        # Conserved quantities
        l = np.sum([(I[i] * w0[i]) ** 2 for i in range(3)]) # L^2
        e = np.sum([I[i] * w0[i] ** 2 for i in range(3)]) # 2T
        L = np.sqrt(l)
        # Check ordering
        if (e > l / I[j] and I[i] < I[k]) or (e < l / I[j] and I[i] > I[k]):
            JacobiOrder = False
            I1, I2, I3 = I[k], I[j], I[i]
            w10, w20, w30 = w0[k], - w0[j], w0[i]
        else:
            JacobiOrder = True
            I1, I2, I3 = I[i], I[j], I[k]
            w10, w20, w30 = w0[i], w0[j], w0[k]
        # More useful quantities
        le1 = l - e * I1
        le3 = l - e * I3
        s_w10 = np.sign(w10) if w10 != 0 else 1
        w1m = s_w10 * np.sqrt(le3 / (I1 * (I1 - I3)))
        w2m = - s_w10 * np.sqrt(le3 / (I2 * (I2 - I3)))
        w3m = np.sign(w30) * np.sqrt(le1 / (I3 * (I3 - I1)))
        wp = np.sign(I2 - I3) * np.sign(w30) * np.sqrt(le1 * (I3 - I2) / np.prod(I))
        m = le3 * (I1 - I2) / (le1 * (I3 - I2))
        epsilon = sp.ellipkinc(np.arcsin(w20 / w2m), m)
        # Evolution in time
        sn, cn, dn, _ = sp.ellipj(wp * t + epsilon, m)
        w1, w2, w3 = w1m * cn, w2m * sn, w3m * dn
        w = np.empty((3, len(t)))
        if JacobiOrder:
            w[i], w[j], w[k] = w1, w2, w3
        else:
            w[i], w[j], w[k] = w3, - w2, w1
        # Align with invariant direction
        Lperp0 = np.sqrt((I1 * w10) ** 2 + (I2 * w20) ** 2)
        e10 = np.array([I1 * I3 * w10 * w30 / (L * Lperp0), - I2 * w20 / Lperp0, I1 * w10 / L])
        e20 = np.array([I2 * I3 * w20 * w30 / (L * Lperp0), I1 * w10 / Lperp0, I2 * w20 / L])
        e30 = np.array([- Lperp0 / L, 0, I3 * w30 / L])
        if JacobiOrder:
            T1t0 = np.array([e10, e20, e30]).T
        else:
            T1t0 = np.array([e30, - e20, e10]).T
        # Permutation of coords
        if j == 1:
            U = np.eye(3)
        elif j == 2:
            U = np.array([
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0]
            ])
        else:
            U = np.array([
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ])
        # More useful quantities
        B = T1t0 @ U.T @ A0 
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
        Re_theta, Im_theta = np.zeros((2, len(t)))
        for n in range(NT):
            M = (2 * n + 1) * np.pi * (wp * t + epsilon) / (2 * K)
            Re_theta += cr[n] * np.sin(M)
            Im_theta += ci[n] * np.cos(M)
        C = np.cos(A1 + A2 * t)
        S = np.sin(A1 + A2 * t)
        theta = np.linalg.norm([Re_theta, Im_theta], axis=0)
        cos_psi = (C * Re_theta + S * Im_theta) / theta
        sin_psi = (S * Re_theta - C * Im_theta) / theta
        Lperp = np.sqrt((I1 * w1) ** 2 + (I2 * w2) ** 2)
        e1 = np.array([I1 * I3 * w1 * w3 / (L * Lperp), - I2 * w2 / Lperp, I1 * w1 / L])
        e2 = np.array([I2 * I3 * w2 * w3 / (L * Lperp), I1 * w1 / Lperp, I2 * w2 / L])
        e3 = np.array([- Lperp / L, np.zeros(len(t)), I3 * w3 / L])
        if JacobiOrder:
            T1 = np.array([e1, e2, e3]) # 3x3xn
        else:
            T1 = np.array([e3, - e2, e1]) # 3x3xn
        z, o = np.zeros(len(t)), np.ones(len(t))
        T2 = np.array([
                [cos_psi, sin_psi, z],
                [- sin_psi, cos_psi, z],
                [z, z, o]
            ]) # 3x3xn
        A = np.einsum('ij,jkl,kml,mn->inl', U, T1, T2, B) # 3x3xn
        return w, A