import ot
import RJW as rjw
import numpy as np
from numpy.linalg import matrix_power
import time
from graph import NoAttrMatrix
from utils import hamming_dist
from IPython.core.debugger import Tracer

"""
The following classes adapt the OT distances to Graph objects
"""

class BadParameters(Exception):
    pass

class RJW_distance():
    """ 
    Attributes
    ----------  
    alpha : float 
            The alpha parameter of RJW
    method : string
             The name of the method used to compute the structures matrices of the graphs.
    max_iter : integer
               Number of iteration of the FW algorithm for the computation of RJW.
    features_metric : string
                      The name of the method used to compute the cost matrix between the features
    transp : ndarray, shape (ns,nt) 
           The transport matrix between the source distribution and the target distribution
    amijo : bool, optionnal
            If True the steps of the line-search is found via an amijo research. Else closed form is used.  
            If there is convergence issues use False.
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.
    """

    def __init__(self,alpha=0.5,beta=0.5,method='random_walk',features_metric='sqeuclidean',max_iter=500,
                 verbose=False,amijo=True):  
        self.method=method
        self.max_iter=max_iter
        self.alpha=alpha
        self.beta=beta
        self.features_metric=features_metric
        self.transp=None
        self.log=None
        self.verbose=verbose
        self.amijo=amijo

    def reshaper(self,x):
        try:
            a=x.shape[1]
            return x
        except IndexError:
            return x.reshape(-1,1)

    def calc_rjw(self,M,C1,C2,L1,L2,E1,E2,D_dist1,D_dist2,t1masses,t2masses):
        transp_rjw,log= rjw.rjw_lp(M,C1,C2,L1,L2,E1,E2,D_dist1,D_dist2,t1masses,t2masses,
                                  loss_fun='square_loss',
                                  alpha=self.alpha,
                                  beta=self.beta,
                                  amijo=self.amijo,G0=None,log=True)
        
        return transp_rjw,log
        
    def graph_d(self,graph1,graph2):
        """ Compute the Wasserstein distance between two graphs. Uniform weights are used.        
        Parameters
        ----------
        graph1 : a Graph object
        graph2 : a Graph object
        Returns
        -------
        The Wasserstein distance between the local variations of graph signals on graph1 and graph2
        """
        gofeature=True
        nodes1=graph1.nodes()
        nodes2=graph2.nodes()
        startstruct=time.time()
        C1,L1,E1,D_dist1=graph1.distance_matrix(method=self.method)
        C2,L2,E2,D_dist2=graph2.distance_matrix(method=self.method)
        end2=time.time()
        t1masses = np.ones(len(nodes1))/len(nodes1)
        t2masses = np.ones(len(nodes2))/len(nodes2)
        try :
            x1=self.reshaper(graph1.all_matrix_attr())
            x2=self.reshaper(graph2.all_matrix_attr())
        except NoAttrMatrix:
            x1=None
            x2=None
            gofeature=False
        if gofeature : 
            if self.features_metric=='dirac':
                f=lambda x,y: x!=y
                M=ot.dist(x1,x2,metric=f)
            elif self.features_metric=='hamming_dist':
                #Eigenvalues of normalized laplacian lie within [0, 2], thus, its maximum eigenvalue is 2.
                tvg1 = np.abs(x1 - (matrix_power(L1, 2).dot(x1)/2))
                tvg2 = np.abs(x2 - (matrix_power(L2, 2).dot(x2)/2))
                
                #Concatenate original and local variational signals
                tvg1 = np.concatenate((x1,tvg1), axis=1)
                tvg2 = np.concatenate((x2,tvg2), axis=1)
                
                f=lambda x,y: hamming_dist(x,y)
                M=ot.dist(tvg1,tvg2,metric=f)
            elif self.features_metric=='euclidean':
                tvg1 = np.abs(x1 - (matrix_power(L1, 2).dot(x1)/2))
                tvg2 = np.abs(x2 - (matrix_power(L2, 2).dot(x2)/2))
                
                #Concatenate original and local variational signals
                tvg1 = np.concatenate((x1,tvg1), axis=1)
                tvg2 = np.concatenate((x2,tvg2), axis=1)
                
                M=ot.dist(tvg1,tvg2,metric=self.features_metric)
            else:
                M=ot.dist(x1,x2,metric=self.features_metric)
            self.M=M
        else:
            M=np.zeros((C1.shape[0],C2.shape[0]))

        startdist=time.time()
        transp_rjw,log=self.calc_rjw(M,C1,C2,L1,L2,E1,E2,D_dist1,D_dist2,t1masses,t2masses)
        enddist=time.time()

        enddist=time.time()
        log['struct_time']=(end2-startstruct)
        log['dist_time']=(enddist-startdist)
        self.transp=transp_rjw
        self.log=log

        return log['loss'][::-1][0]

    def get_tuning_params(self):
        """Parameters that defined the RJW distance """
        return {"method":self.method,"max_iter":self.max_iter,"alpha":self.alpha,
        "features_metric":self.features_metric,"amijo":self.amijo}
