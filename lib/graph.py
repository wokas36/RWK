import networkx as nx
import matplotlib.pyplot as plt
import itertools
import numpy as np
import ot
import time
from scipy import sparse
import copy
import matplotlib.colors as mcol
from matplotlib import cm
from attentionwalk import AttentionWalkTrainer
from scipy.linalg import fractional_matrix_power
from IPython.core.debugger import Tracer

class NoAttrMatrix(Exception):
    pass

class NoPathException(Exception):
    pass

"""
Summarizes all the methods and classes related to graphs
"""

class Graph():
    """ Graph is a class that model all the graphs used in the experiments.
    
    Attributes
    ----------    
    nx_graph : a networkx graph, optionnal
               The networkx graph
    C : ndarray
        The structure matrix of the graph. Initalize at None
    name_struct_dist : string
                       The name of the method used to compute the structure matrix
    name : string, 
           Name of the graph because life without name has no meaning. 
    """
    def __init__(self,nx_graph=None):
        if nx_graph is not None:
            self.nx_graph=nx.Graph(nx_graph)
        else:
            self.nx_graph=nx.Graph()
        self.name='A graph as no name'
        self.log={}
        self.log['pertoperdistance']=[]
        self.log['pathtime']=[]
        self.log['attridist']=[]
        self.C=None
        self.L=None
        self.E=None
        self.degree_distribution=None
        self.name_struct_dist='No struct name for now'


    def __eq__(self, other) : 
        return self.nx_graph == other.nx_graph

    def __hash__(self):
        return hash(str(self))

    def characterized(self):
        if self.name!='A graph as no name':
            return self.name
        else:
            return self

    def nodes(self):
        """ returns the vertices of a graph """
        return dict(self.nx_graph.nodes())

    def edges(self):
        """ returns the edges of a graph """
        return self.nx_graph.edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.nodes():
            self.nx_graph.add_node(vertex)

    def values(self):
        """ returns a list of all the features of the graph"""
        return [v for (k,v) in nx.get_node_attributes(self.nx_graph,'attr_name').items()]

    def add_nodes(self, nodes):
        self.nx_graph.add_nodes_from(nodes)

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        (vertex1, vertex2) = tuple(edge)
        self.nx_graph.add_edge(vertex1,vertex2)

    def add_one_attribute(self,node,attr,attr_name='attr_name'):
        self.nx_graph.add_node(node,attr_name=attr)

    def add_attibutes(self,attributes):
        attributes=dict(attributes)
        for node,attr in attributes.items():
            self.add_one_attribute(node,attr)

    def get_attr(self,vertex):
        return self.nx_graph.node[vertex]
    
    def reshaper(self,x):
        try:
            a=x.shape[1]
            return x
        except IndexError:
            return x.reshape(-1,1)

    def distance_matrix(self,method='random_walk',changeInf=True,maxvaluemulti=10,force_recompute=False): 
        """ Compute the structure matrix of the graph. 
        It aims at comparing nodes between them using a notion of similarity defined by the "method" parameter
        
        Parameters
        ----------
        method : string, default random_walk. choices : random_walk and adjency
               The method used to compute the structure matrix of the graph :
                   - random_walk : compute embeddings from random walk between the nodes
                   - adjency : compute the adjency matrix of the graph
        changeInf : bool
                    If true when the graph has disconnected parts it replaces inf distances by a maxvaluemulti 
                    times the largest value of the structure matrix
        force_recompute : force to recompute de distance matrix. If False the matrix is computed only if not 
        already compute or if the method used for computing it changes       
        Returns
        -------
        C : ndarray, shape (n_nodes,n_nodes)
            The structure matrix of the graph
        Set also the attribute C of the graph if C does not exist or if force_recompute is True 
        """
        start=time.time()
        if (self.C is None) or force_recompute:

            A=nx.adjacency_matrix(self.nx_graph)
                
            if method=='adjency':
                return A.toarray()
                
            if method=='random_walk':
                
                #parameters for random walk embedding
                dimensions = 64
                epochs = 200
                window_size = 6
                num_of_walks = 20
                beta = 0.5
                gamma = 0.5
                learning_rate = 0.01
                G = self.nx_graph
                
                random_walk_model = AttentionWalkTrainer(G, dimensions, epochs, window_size, num_of_walks, 
                                                       beta, gamma, learning_rate)
                random_walk_model.fit()
                random_walk_emb_adj = random_walk_model.create_embedding()
                
                C = ot.dist(random_walk_emb_adj, random_walk_emb_adj, metric='euclidean')
                self.E = random_walk_emb_adj

            if changeInf==True:
                C[C==float('inf')]=maxvaluemulti*np.max(C[C!=float('inf')])
                
            #construct normalized laplacian.
            adj = A.toarray()
            #Degree matrix.
            degree = np.sum(adj, axis=0)
            degree_distribution = degree/np.linalg.norm(degree, ord=1)
            #Tracer()()
            degree = np.matrix(np.diag(degree))
            I = np.matrix(np.eye(adj.shape[0]))
            #Laplacian matrix.
            normalized_L = I - (fractional_matrix_power(degree, -0.5) * adj * fractional_matrix_power(degree, -0.5))
            
            self.C=C
            self.L=normalized_L
            self.degree_distribution=degree_distribution
            self.name_struct_dist=method
            end=time.time()
            self.log['allStructTime']=(end-start)
            return self.C, self.L, self.E, self.degree_distribution

        else :
            end=time.time()
            self.log['allStructTime']=(end-start)
            return self.C, self.L, self.E, self.degree_distribution


    def all_matrix_attr(self,return_invd=False):
        d=dict((k, v) for k, v in self.nx_graph.nodes.items())
        x=[]
        invd={}
        try :
            j=0
            for k,v in d.items():
                x.append(v['attr_name'])
                invd[k]=j
                j=j+1
            if return_invd:
                return np.array(x),invd
            else:
                return np.array(x)
        except KeyError:
            raise NoAttrMatrix
            

def relabel_graph_order(graph):

    relabel_dict_={}
    graph_node_list=list(graph.nodes())
    for i in range(len(graph_node_list)):
        relabel_dict_[graph_node_list[i]]=i
        i+=1

    inv_relabel_dict_={v:k for k,v in relabel_dict_.items()}

    graph_relabel=nx.relabel_nodes(graph,relabel_dict_)

    return graph_relabel,inv_relabel_dict_

def wl_labeling(graph,h=2,tohash=True):
    """ Computes the Weisfeler-Lehman labeling for all nodes
    Parameters
    ----------
    graph : Graph
            The Graph to relabel
    h : integer
          The number of iteration of the Weisfeler-Lehman coloring.
    tohash : bool, optionnal
          Wether to hash the concatenated labeled
    Returns
    -------
    graphs : Graph, 
        The relabeled graph

    References
    ----------
    .. [1] Nils M. Kriege and Pierre{-}Louis Giscard and Richard C. Wilson
        "On Valid Optimal Assignment Kernels and Applications to Graph Classification"
        Advances in Neural Information Processing Systems 29 (NIPS). 2016.

    """
    niter=1
    final_graph=nx.Graph(graph)

    graph_relabel,inv_relabel_dict_=relabel_graph_order(final_graph)
    l_aux = list(nx.get_node_attributes(graph_relabel,'attr_name').values())
    labels = np.zeros(len(l_aux), dtype=np.int32)

    adjency_list = list([list(x[1].keys()) for x in graph_relabel.adjacency()]) 

    for j in range(len(l_aux)):
        labels[j] = l_aux[j]

    new_labels = copy.deepcopy(l_aux)

    while niter<=h:

        labeled_graph=nx.Graph(final_graph)

        graph_relabel,inv_relabel_dict_=relabel_graph_order(final_graph)

        l_aux = list(nx.get_node_attributes(graph_relabel,'attr_name'+str(niter-1)).values())

        adjency_list = list([list(x[1].keys()) for x in graph_relabel.adjacency()]) 

        for v in range(len(adjency_list)):

            prev_neigh=np.sort([labels[adjency_list[v]]][-1])

            long_label = np.concatenate((np.array([[labels[v]][-1]]),prev_neigh))
            long_label_string = ''.join([str(x) for x in long_label])
            new_labels[v] =long_label_string

        labels = np.array(copy.deepcopy(new_labels))

        dict_={inv_relabel_dict_[i]:labels[i] for i in range(len(labels))}

        nx.set_node_attributes(labeled_graph,dict_,'attr_name'+str(niter))
        niter+=1
        final_graph=nx.Graph(labeled_graph)

    dict_values={}
    for k,v in final_graph.nodes().items():
        hashed=sorted([str(x) for x in v.values()], key=len)

        if tohash :
            dict_values[k]=np.array([hash(x) for x in hashed])
        else:
            dict_values[k]=np.array(hashed)

    graph2=nx.Graph(graph)
    nx.set_node_attributes(graph2,dict_values,'attr_name')

    return graph2