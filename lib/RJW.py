import numpy as np
import ot
import scg_optimizer
from utils import dist,reshaper,hamming_dist
from scipy import stats
from scipy.sparse import random
from IPython.core.debugger import Tracer

class StopError(Exception):
    pass
      
def init_matrix(C1,C2,p,q,loss_fun='square_loss'):
    """ Return loss matrices and tensors for Gromov-Wasserstein fast computation
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
    function as the loss function of Gromow-Wasserstein discrepancy.
    The matrices are computed as described in Proposition 1 in [1]
    Where :
        * C1 : Metric cost matrix in the source space
        * C2 : Metric cost matrix in the target space
        * T : A coupling between those two spaces
    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=(a^2)
            * f2(b)=(b^2)
            * h1(a)=a
            * h2(b)=2b
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
         Coupling between source and target spaces
    p : ndarray, shape (ns,)
    Returns
    -------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
            
    if loss_fun == 'square_loss':
        def f1(a):
            return a**2 

        def f2(b):
            return b**2

        def h1(a):
            return a

        def h2(b):
            return 2*b
        
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC=constC1+constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC,hC1,hC2

def tensor_product(constC,hC1,hC2,T):

    """ Return the tensor for Gromov-Wasserstein fast computation
    The tensor is computed as described in Proposition 1 Eq. (6) in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
    
    A=-np.dot(hC1, T).dot(hC2.T)
    tens = constC+A

    return tens

def gwloss(constC,hC1,hC2,T):

    """ Return the Loss for Gromov-Wasserstein
    The loss is computed as described in Proposition 1 Eq. (6) in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    loss : float
           Gromov Wasserstein loss
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    tens=tensor_product(constC,hC1,hC2,T) 
              
    return np.sum(tens*T) 

def lpreg1(E, L, G):
    EG = np.dot(E.T, G)
    product = np.dot(EG, L).dot(EG.T)
    #product = np.dot(E.T, G).dot(L).dot(G.T).dot(E)
    return np.trace(product)

def lpreg2(E, L, G):
    GE = np.dot(G, E)
    product = np.dot(GE.T, L).dot(GE)
    #product = np.dot(E.T, G.T).dot(L).dot(G).dot(E)
    return np.trace(product)

def gwggrad(constC,hC1,hC2,T):
    
    """ Return the gradient for Gromov-Wasserstein
    The gradient is computed as described in Proposition 2 in [1].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    grad : ndarray, shape (ns, nt)
           Gromov Wasserstein gradient
    References
    ----------
    .. [1] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
          
    return 2*tensor_product(constC,hC1,hC2,T) 

def lpgrad1(E, L, G):
    product1 = np.dot(E, G).dot(L.T)
    product2 = np.dot(E, G).dot(L)
    #product1 = np.dot(E, E.T).dot(G).dot(L.T)
    #product2 = np.dot(E, E.T).dot(G).dot(L)
    return product1 + product2

def lpgrad2(E, L, G):
    product1 = np.dot(L.T, G).dot(E)
    product2 = np.dot(L, G).dot(E)
    #product1 = np.dot(L.T, G).dot(E).dot(E.T)
    #product2 = np.dot(L, G).dot(E).dot(E.T)
    return product1 + product2
    
def rjw_lp(M,C1,C2,L1,L2,E1,E2,D_dist1,D_dist2,p,q,loss_fun='square_loss',alpha=1,beta=1e-3,amijo=True,G0=None,**kwargs): 
    """
    Computes the RW distance between two graphs
    Parameters
    ----------
    M  : ndarray, shape (ns, nt)
         Metric cost matrix between features across domains
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix respresentative of the structure in the source space
    C2 : ndarray, shape (nt, nt)
         Metric cost matrix espresentative of the structure in the target space
    L1 : ndarray, shape (ns, ns)
         Laplacian matrix of source graph
    L2 : ndarray, shape (nt, nt)
         Laplacian matrix of target graph
    E1 : ndarray, shape (ns, d)
         Random walk embbedings of source graph
    E2 : ndarray, shape (nt, d)
         Random walk embbedings of target graph
    D_dist1 : ndarray, shape (ns,)
         Degree distribution of source graph
    D_dist2 : ndarray, shape (nt,)
         Degree distribution of target graph
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string,optionnal
        loss function used for the solver 
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    amijo : bool, optional
        If True the steps of the line-search is found via an amijo research. Else closed form is used.
        If there is convergence issues use False.
    **kwargs : dict
        parameters can be directly pased to the gcg solver
    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    """
    f=lambda x,y: hamming_dist(x,y)
    M = M + 1e-3 * ot.dist(E1, E2, metric=f)
    
    #D_dist1 = np.reshape(D_dist1, (-1,1))
    #D_dist2 = np.reshape(D_dist2, (-1,1))
    
    #joint_distribution=D_dist1*D_dist2.T
    
    joint_distribution = np.zeros((D_dist1.shape[0], D_dist2.shape[0]))
    
    #calculate the |D_dist1[i]-D_dist2[j]| to get the joint degree distribution.
    for i in np.arange(D_dist1.shape[0]):
        for j in np.arange(D_dist2.shape[0]):
            normalized_factor = np.maximum(D_dist1[i],D_dist2[j])
            joint_distribution[i][j] = 1-(np.abs(D_dist1[i]-D_dist2[j])/normalized_factor)
            
    joint_distribution = joint_distribution/np.linalg.norm(joint_distribution.sum(axis=1), ord=1)
    
    E1_inner = np.dot(E1, E1.T)
    E2_inner = np.dot(E2, E2.T)

    constC,hC1,hC2=init_matrix(C1,C2,p,q,loss_fun)
    
    if G0 is None:
        G0=p[:,None]*q[None,:]
    
    def f1(G):
        return gwloss(constC,hC1,hC2,G)
    def f2(G):
        return lpreg1(E1, L2, G)
    def f3(G):
        return lpreg2(E2, L1, G)
    def df1(G):
        return gwggrad(constC,hC1,hC2,G)
    def df2(G):
        return lpgrad1(E1_inner, L2, G)
    def df3(G):
        return lpgrad2(E2_inner, L1, G)
 
    return scg_optimizer.scg(p, q, M, alpha, 1e-3, 1e-3, beta, f1, f2, f3, df1, df2, df3, joint_distribution, G0, amijo=amijo, 
                     C1=C1, C2=C2, constC=constC, **kwargs)