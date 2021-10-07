"""
Optimization algorithms for OT
"""
import numpy as np
from scipy.optimize.linesearch import scalar_search_armijo
from ot.lp import emd
from sinkhorn_algorithms import sinkhorn
from IPython.core.debugger import Tracer

class StopError(Exception):
    pass

class NonConvergenceError(Exception):
    pass
class StopError(Exception):
    pass
        
def line_search_armijo(f, xk, pk, gfk, old_fval, args=(), c1=1e-4, alpha0=0.99):
    """
    Armijo linesearch function that works with matrices
    find an approximate minimum of f(xk+alpha*pk) that satifies the
    armijo conditions.
    Parameters
    ----------
    f : function
        loss function
    xk : np.ndarray
        initial position
    pk : np.ndarray
        descent direction
    gfk : np.ndarray
        gradient of f at xk
    old_fval : float
        loss value at xk
    args : tuple, optional
        arguments given to f
    c1 : float, optional
        c1 const in armijo rule (>0)
    alpha0 : float, optional
        initial step (>0)
    Returns
    -------
    alpha : float
        step that satisfy armijo conditions
    fc : int
        nb of function call
    fa : float
        loss value at step alpha
    """
    xk = np.atleast_1d(xk)
    fc = [0]

    def phi(alpha1):
        fc[0] += 1
        return f(xk + alpha1 * pk, *args)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval

    derphi0 = np.sum(gfk.T * pk)
    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1, alpha0=alpha0)

    return alpha, fc[0], phi1

    
def scg(a, b, M, reg1, reg2, reg3, beta, f1, f2, f3, df1, df2, df3, j_dist, G0=None, numItermax=10, numInnerItermax=50, 
        stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False,amijo=True, C1=None, C2=None, constC=None):
    """
    PTC-MR AND ENZYMES -> numItermax=5, numInnerItermax=20
    MUTAG, BZR AND COX2 -> numItermax=10, numInnerItermax=50
    PROTEINS -> numItermax=3, numInnerItermax=50
    
    Solve the general regularized OT problem with the sinkhorn conditional gradient

    - M is the (ns, nt) metric cost matrix
    - a and b are source and target weights (sum to 1)

    Parameters
    ----------
    a : ndarray, shape (ns, )
        samples weights in the source domain
    b : ndarrayv (nt, )
        samples in the target domain
    M : ndarray, shape (ns, nt)
        loss matrix
    reg1 : float
        Entropic Regularization term >0
    reg2 : float
        Second Regularization term >0 (target regularization)
    reg3 : float
        Third Regularization term >0 (source regularization)
    beta: float
        Penalty term > 0 (rho regularization)
    f1 : g(\gamma) function  
        Gromov Wasserstein loss
    f2 : Regularization function
        Target regularization
    f3 : Regularization function 
        Source regularization
    df1 : Gradient function
        Gradient of Gromov Wasserstein loss
    df2 : Gradient function
        Gradient of target regularization
    df3 : Gradient function
        Gradient of source regularization
    j_dist : ndarray, shape (ns, nt)
        Joint degree distribution
    G0 : ndarray, shape (ns, nt), optional
        initial guess (default is indep joint density)
    numItermax : int, optional
        Max number of iterations
    numInnerItermax : int, optional
        Max number of iterations of Sinkhorn
    stopThr : float, optional
        Stop threshol on the relative variation (>0)
    stopThr2 : float, optional
        Stop threshol on the absolute variation (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : ndarray, shape (ns, nt)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    """

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = np.outer(a, b)
    else:
        G = G0

    def cost(G):
        return np.sum(M * G) + reg2 * f2(G) + reg3 * f3(G) + beta * np.linalg.norm(G,'fro')**2 + reg1 * f1(G) - 1e-4 * (np.sum(G * np.log(G)) - np.sum(G * np.log(j_dist)))

    f_val = cost(G)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, 0, 0))

    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg1 * df1(G)
        # set M positive
        Mi += Mi.min()

        # solve linear program with Sinkhorn-knopp
        # MUTAG, PTC-MR, COX2 AND BZR -> 0.5
        # ENZYMES AND PROTEINS -> 0.9
        Gc = sinkhorn(a, b, Mi, 0.5, method='sinkhorn', numItermax=numInnerItermax)

        deltaG = Gc - G

        # line search
        dcost = Mi + reg2 * df2(G) + reg3 * df3(G) + beta * G - 1e-4 * (1 + np.log(G) - np.log(j_dist))
        # set dcost positive
        dcost += dcost.min()
        alpha, fc, f_val = line_search_armijo(cost, G, deltaG, dcost, f_val)

        if alpha is None:
            print(it)
        if alpha is None or np.isnan(alpha) :
            raise NonConvergenceError('Alpha is not converged')
        else:
            G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        
        # computing suboptimality gap by Frobenius inner product
        #delta_i = np.multiply(dcost.T, (G - Gc)).sum()
        delta_i = np.trace(dcost.T @ (G - Gc))

        if delta_i <= stopThr or abs_delta_fval <= stopThr2:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val, relative_delta_fval, abs_delta_fval))

    if log:
        return G, log
    else:
        return G