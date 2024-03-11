import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad
from scipy.sparse import lil_matrix
from scipy.linalg import eigh
from time import time
from icecream import ic


def hermite(n_nodes, n_diff, plot=False):
    """
    Compute Hermite interpolation polynomial basis set for use
    in the Hermite FEM method.
    
    The function computes the coefficients of the Hermite
    interpolation polynomials for a given number of nodes and
    a given number of derivatives. The coefficients are computed
    by solving the linear system of equations that arises from
    the interpolation conditions.
    
    The matrix C returned contains the coefficients of the
    resulting polynommials. The polynomials are ordered such
    that the first n_nodes polynomials are the interpolating
    polynomials for the function values, the next n_nodes polynomials
    are interpolation polynomials for the first derivative, and so on.
    
    Pass plot=True to plot the resulting polynomials.
    
    Args:
        n_nodes: number of nodes
        n_diff: number of derivatives
        plot: if True, plot the resulting polynomials
    Returns:
        dict: {'n_nodes': n_nodes, 'n_diff': n_diff, 'C': C}, where C = matrix of coefficients.
    """
    
    m = n_diff + 1
    xi = np.linspace(-1,1,n_nodes)
    n = n_nodes*m
    A = np.zeros((n, n))
    # (d/dx)^j p_{i,j'}(x_i') = \delta_{i,i'}\delta_{j,j'}
    for j in range(n_nodes):
        for beta in range(m):
            for k in range(beta, n):
                factor = np.exp(gammaln(k+1)-gammaln(k-beta+1))
                #print(factor, k, beta, j, x[j])
                A[beta*n_nodes+j,k] = factor * np.power(xi[j],k - beta)

    #print('A = ')
    #print(A)                 
    C = np.linalg.inv(A)
    #print('C = ', C)
    
    if plot:
        x = np.linspace(-1,1,200)
        plt.figure()
        plt.plot(xi, 0*xi, 'o')
        for j in range(n_nodes):
            for beta in range(m):
                p = C[:,beta*n_nodes+j]
                y = np.zeros_like(x)
                for k in range(len(p)):
                    y += p[k]*x**k
                style = f'C{beta}-'
                plt.plot(x,y,style, label=f'j={j}, beta={beta}')

        plt.legend()
        plt.title('Hermite interpolation polynomials')
        plt.xlabel('x')
        plt.ylabel(r'$p^\beta_j(x)$')   
        plt.show()                
            
    return {'n_nodes': n_nodes, 'n_diff': n_diff, 'C': C}


def hermite_fem_matrix(x_el, basis, difforder=[0,0], weight=lambda x: np.ones_like(x)):
    """
    Assemble a finite element matrix for Hermite basis functions.
    

    Args:
        x_el (array-like): Array of element boundaries.
        basis (dict): Dictionary containing basis function information.
        difforder (list, optional): List of differentiation orders for the basis functions. Defaults to [0, 0].
        weight (function, optional): Weighting function for the integration. Defaults to lambda x: np.ones_like(x).

    Returns:
        csr_matrix: Finite element matrix.

    Raises:
        AssertionError: If the number of degrees of freedom does not match the length of the basis function coefficients.

    """
        
    n_nodes_per_el = basis['n_nodes']
    n_diff = basis['n_diff']
    C = basis['C'] # coeffs of element basis functions
    n_beta = n_diff + 1
    n_dof = n_nodes_per_el*n_beta
    n_el = len(x_el) - 1
    
    # set up FEM nodes
    # these are 1:1 with the global basis functions.
    x = np.zeros((n_nodes_per_el-1)*n_el + 1)
    n_nodes = len(x)
    for e in range(n_el):
        x0 = x_el[e]
        x1 = x_el[e+1]
        x[e*(n_nodes_per_el-1):(e+1)*(n_nodes_per_el-1)+1] = np.linspace(x0, x1, n_nodes_per_el)
    
    dim = n_nodes*n_beta
#    ic(n_nodes, n_el, x)
  
    basis['x'] = x   
  
    def glob_to_loc(I):
        version = 2

        if version == 1:
            # version 1 has a
            e = I // (n_beta*(n_nodes_per_el-1))
            i0 = I % (n_beta*(n_nodes_per_el-1))
            i = i0 % (n_nodes_per_el-1)
            beta = i0 // (n_nodes_per_el-1)
            
        elif version == 2:        
            e = I // (n_beta*(n_nodes_per_el-1))
            i0 = I % (n_beta*(n_nodes_per_el-1))
            beta = i0 % n_beta
            i = i0 // n_beta
        elif version == 3:
            beta = I // (n_nodes*n_beta)
            i0 = I % (n_nodes*n_beta)
            i = i0 // n_el
            e = i0 % n_el

        #ic(e, i)

        # get the local basis functions that contribute to 
        # the global basis function I.
        if i == 0 and e > 0 and e < n_el:
            e_set = [e-1, e] 
            i_set = [n_nodes_per_el-1,0]
            beta_set = [beta, beta]
        elif i == 0 and e == 0:
            e_set = [e]
            i_set = [0]
            beta_set = [beta]
        elif i == 0 and e == n_el:
            e_set = [e-1]
            i_set = [n_nodes_per_el-1]
            beta_set = [beta]
        else:
            e_set = [e]
            i_set = [i]
            beta_set = [beta]
            
        return e_set, i_set, beta_set
              
    # loop over global basis functions.
    S = lil_matrix((dim, dim))
    
    for I in range(dim):

        #ic(e, i)

        # get the local basis functions that contribute to 
        # the global basis function I.
        e_set, i_set, beta_set = glob_to_loc(I)
        #ic(e_set, i_set, beta_set)
        #ic(e_set, i_set)
        for J in range(dim):
            e_set2, i_set2, beta_set2 = glob_to_loc(J)
            #ic(e_set2, i_set2,beta_set2)
                        
            # check if e_set and e_set2 have common elements.
            # for each common element, compute local matrix element and add
            # to global matrix.
            for e1,i1,beta1 in zip(e_set,i_set,beta_set):
                for e2,i2,beta2 in zip(e_set2,i_set2,beta_set2):
                    if e1 == e2:
                        # ic(e1, len(x_el))
                        #print(f'I={I}, J={J}, overlap between e={e1},i={i1} and e={e2},i={i2}, beta={beta1}, beta={beta2}')
                        p1 = np.flip(C[:,beta1*n_nodes_per_el+i1])
                        p2 = np.flip(C[:,beta2*n_nodes_per_el+i2])
                        #if difforder[0] > 0:
                        p1 = np.polyder(p1, difforder[0])
                        #if difforder[1] > 0:
                        p2 = np.polyder(p2, difforder[1])
                        h_e = x_el[e1+1]-x_el[e1]
                        jac = np.power(h_e / 2, difforder[0] + difforder[1])
                        p1_xi = lambda xi: np.polyval(p1, xi)
                        p2_xi = lambda xi: np.polyval(p2, xi)
                        xx = lambda xi: x_el[e1] + (1+xi)*h_e/2
                        integrand = lambda xi: p1_xi(xi)*p2_xi(xi)*weight(xx(xi))
                        temp = fixed_quad(integrand, -1, 1, n = 10)[0]
                        #ic(temp)
                        S[I,J] += temp * (x_el[e1+1]-x_el[e1]) / 2 / jac
                        
                        #pass

          
    return S.tocsr()
            


if __name__ == '__main__':


    
    basis = hermite(3, 1, plot=True)
    x_el = np.linspace(0,10,40)
    tic = time()
    S = hermite_fem_matrix(x_el, basis, difforder=[0,0], weight = lambda x: x)
    T = .5 * hermite_fem_matrix(x_el, basis, difforder=[1,1], weight = lambda x: x)
    V = hermite_fem_matrix(x_el, basis, difforder=[0,0], weight = lambda x: -1.0)
    toc = time()
    ic(toc-tic)
