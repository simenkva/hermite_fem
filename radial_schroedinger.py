
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from fem_1d import hermite_fem_matrix, hermite
from icecream import ic


def radial_hydrogen(r_el, p = 1, lump = False):

    # set up standard "P elements", with p
    # being the degree of the polynomial,
    # equivalently the number of nodes per element.
    basis = hermite(p+1, 0, plot=False)
    
    weight = lambda r: r**2
    T = .5 * hermite_fem_matrix(r_el, basis, difforder=[1,1], weight = weight)
    S = hermite_fem_matrix(r_el, basis, difforder=[0,0], weight = weight)
    V = -1.0 * hermite_fem_matrix(r_el, basis, difforder=[0,0], 
                           weight = lambda r: r)

    if not lump:
        H = T + V
        E, U = eigh(H.toarray()[:-1,:-1], S.toarray()[:-1,:-1])
    else:
        H = (T + V).toarray()
        S = S.toarray()
        S_diag = np.sum(S, axis = 1)
        ic(S.shape)
        H = np.diag(S_diag**(-.5)) @ H @ np.diag(S_diag**(-.5))
        E, U = eigh(H[:-1,:-1])
        
    return E, U, basis
    
    
# set up a grid.

# uniform grid
#r_el = np.linspace(0, 100, 100)

# non-uniform grid
r_el = np.linspace(0, 10, 100)**2
ic(r_el.shape)

# compute the eigenvalues and eigenvectors.
E, U, basis = radial_hydrogen(r_el, p = 2, lump=False)
print('A few lowest eigenvalues:', E[:5])

# plot the lowest eigenfunctions.
r = basis['x'][:-1]
plt.figure()
plt.plot(r, U[:,0], '*-')
plt.plot(r, U[:,1], '*-')
plt.show()