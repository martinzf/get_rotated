import numpy as np
import scipy.special as sp

def solve(I, w0, t):
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
    return w