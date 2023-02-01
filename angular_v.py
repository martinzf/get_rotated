import numpy as np
import scipy.special as sp

def solve(I, w0, t):
    if (np.count_nonzero(w0 == 0) == 2) or (len(set(I)) == 1): # w along principal axis or spherical symmetry
        w0 = np.reshape(w0, (3, 1))
        return np.tile(w0, len(t))
    elif len(set(I)) == 2: # Axially symmetric
        # Set 3rd axis as axis corresponding to different moment of inertia
        k = np.argmax(np.abs(I - np.median(I)))
        i = (k + 1) % 3
        j = (k + 2) % 3
        if w0[k] == 0: # w along axis perpendicular to symmetry axis
            w0 = np.reshape(w0, (3, 1))
            return np.tile(w0, len(t))
        # Precession of w around symmetry axis
        O = (I[k] / I[i] - 1) * w0[k]
        c, s = np.cos(O * t), np.sin(O * t)
        w = np.empty((3, len(t)))
        w[i] = c * w0[i] - s * w0[j]
        w[j] = s * w0[i] + c * w0[j]
        w[k] = w0[k] * np.ones(len(t))
        return w
    else: # Asymmetric
        # Conserved quantities
        l = np.sum([(I[i] * w0[i]) ** 2 for i in range(3)]) # L^2
        e = np.sum([I[i] * w0[i] ** 2 for i in range(3)]) # 2T
        # Set intermediate axis as axis corresponding to median moment of inertia
        j = np.argsort(I)[len(I) // 2]
        k = (j + 1) % 3
        i = (j + 2) % 3
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
        return w