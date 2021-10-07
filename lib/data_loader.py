from graph import Graph,wl_labeling
import networkx as nx
from utils import per_section,indices_to_one_hot
from collections import defaultdict
import numpy as np
import math
class NotImplementedError(Exception):
    pass

"""
All methods for loading the data
"""

def load_local_data(data_path,name,one_hot=False,attributes=True,use_node_deg=False,wl=0):
    """ Load local datasets    
    Parameters
    ----------
    data_path : string
                Path to the data. Must link to a folder where all datasets are saved in separate folders
    name : string
           Name of the dataset to load. 
           Choices=['mutag','ptc','enzymes','protein','bzr','cox2'] 
    one_hot : integer
              If discrete attributes must be one hotted it must be the number of unique values.
    attributes :  bool, optional
                  For dataset with both continuous and discrete attributes. 
                  If True it uses the continuous attributes (corresponding to "Node Attr." in [5])
    use_node_deg : bool, optional
                   Wether to use the node degree instead of original labels. 
    wl : integer, optional
         For dataset with discrete attributes.
         Relabels the graph with a Weisfeler-Lehman procedure. wl is the number of iteration of the procedure
         See wl_labeling in graph.py
    Returns
    -------
    X : array
        array of Graph objects created from the dataset
    y : array
        classes of each graph    
    References
    ----------    
    [5] Kristian Kersting and Nils M. Kriege and Christopher Morris and Petra Mutzel and Marion Neumann 
        "Benchmark Data Sets for Graph Kernels"
    """
    if name=='mutag':
        path=data_path+'/MUTAG/'
        dataset=build_MUTAG_dataset(path,one_hot=one_hot)
    if name=='ptc':
        path=data_path+'/PTC_MR/'
        dataset=build_PTC_dataset(path,one_hot=one_hot)
    if name=='enzymes':
        path=data_path+'/ENZYMES/'
        if attributes:
            dataset=build_ENZYMES_dataset(path,type_attr='real')
        else:
            dataset=build_ENZYMES_dataset(path)
    if name=='protein':
        path=data_path+'/PROTEINS/'
        if attributes:
            dataset=build_PROTEIN_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_PROTEIN_dataset(path)
    if name=='bzr':
        path=data_path+'/BZR/'
        if attributes:
            dataset=build_BZR_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_BZR_dataset(path)
    if name=='cox2':
        path=data_path+'/COX2/'
        if attributes:
            dataset=build_COX2_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_COX2_dataset(path)
    if name=='nci1':
        path=data_path+'/NCI1/'
        dataset=build_NCI1_dataset(path,one_hot=one_hot)
    if name=='nci109':
        path=data_path+'/NCI109/'
        dataset=build_NCI109_dataset(path,one_hot=one_hot)
    if name=='dd':
        path=data_path+'/DD/'
        dataset=build_DD_dataset(path,one_hot=one_hot)
    X,y=zip(*dataset)
    if wl!=0:
        X=label_wl_dataset(X,h=wl)
    return np.array(X),np.array(y)

#%%

def label_wl_dataset(X,h):
    X2=[]
    for x in X:
        x2=Graph()
        x2.nx_graph=wl_labeling(x.nx_graph,h=2)
        X2.append(x2)
    return X2

#%%

def histog(X,bins=10):
    node_length=[]
    for graph in X:
        node_length.append(len(graph.nodes()))
    return np.array(node_length),{'histo':np.histogram(np.array(node_length),bins=bins),'med':np.median(np.array(node_length))
            ,'max':np.max(np.array(node_length)),'min':np.min(np.array(node_length))}

def node_labels_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=int(elt)
            k=k+1
    return node_dic

def node_attr_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=[float(x) for x in elt.split(',')]
            k=k+1
    return node_dic

def graph_label_list(path,name):
    graphs=[]
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            graphs.append((k,int(elt)))
            k=k+1
    return graphs
    
def graph_indicator(path,name):
    data_dict = defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            data_dict[int(elt)].append(k)
            k=k+1
    return data_dict

def compute_adjency(path,name):
    adjency= defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        for elt in sections[0]:
            adjency[int(elt.split(',')[0])].append(int(elt.split(',')[1]))
    return adjency


def all_connected(X):
    a=[]
    for graph in X:
        a.append(nx.is_connected(graph.nx_graph))
    return np.all(a)

def build_PROTEIN_dataset(path,type_attr='label',use_node_deg=False):
    if type_attr=='label':
        node_dic=node_labels_dic(path,'PROTEINS_node_labels.txt')
    if type_attr=='real':
        node_dic=node_attr_dic(path,'PROTEINS_node_attributes.txt')
    graphs=graph_label_list(path,'PROTEINS_graph_labels.txt')
    adjency=compute_adjency(path,'PROTEINS_A.txt')
    data_dict=graph_indicator(path,'PROTEINS_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_MUTAG_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'MUTAG_graph_labels.txt')
    adjency=compute_adjency(path,'MUTAG_A.txt')
    data_dict=graph_indicator(path,'MUTAG_graph_indicator.txt')
    node_dic=node_labels_dic(path,'MUTAG_node_labels.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],7)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_PTC_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'PTC_MR_graph_labels.txt')
    adjency=compute_adjency(path,'PTC_MR_A.txt')
    data_dict=graph_indicator(path,'PTC_MR_graph_indicator.txt')
    node_dic=node_labels_dic(path,'PTC_MR_node_labels.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],18)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_ENZYMES_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'ENZYMES_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'ENZYMES_node_labels.txt')
    if type_attr=='real':
        node_dic=node_attr_dic(path,'ENZYMES_node_attributes.txt')
    adjency=compute_adjency(path,'ENZYMES_A.txt')
    data_dict=graph_indicator(path,'ENZYMES_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_BZR_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'BZR_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'BZR_node_labels.txt')
    if type_attr=='real':
        node_dic=node_attr_dic(path,'BZR_node_attributes.txt')
    adjency=compute_adjency(path,'BZR_A.txt')
    data_dict=graph_indicator(path,'BZR_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data
    

def build_COX2_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'COX2_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'COX2_node_labels.txt')
    if type_attr=='real':
        node_dic=node_attr_dic(path,'COX2_node_attributes.txt')
    adjency=compute_adjency(path,'COX2_A.txt')
    data_dict=graph_indicator(path,'COX2_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_NCI1_dataset(path,one_hot=False):
    node_dic=node_labels_dic(path,'NCI1_node_labels.txt')
    node_dic2={}
    for k,v in node_dic.items():
        node_dic2[k]=v-1
    node_dic=node_dic2
    graphs=graph_label_list(path,'NCI1_graph_labels.txt')
    adjency=compute_adjency(path,'NCI1_A.txt')
    data_dict=graph_indicator(path,'NCI1_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],37)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_NCI109_dataset(path,one_hot=False):
    node_dic=node_labels_dic(path,'NCI109_node_labels.txt')
    node_dic2={}
    for k,v in node_dic.items():
        node_dic2[k]=v-1
    node_dic=node_dic2
    graphs=graph_label_list(path,'NCI109_graph_labels.txt')
    adjency=compute_adjency(path,'NCI109_A.txt')
    data_dict=graph_indicator(path,'NCI109_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],38)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_DD_dataset(path,one_hot=False):
    node_dic=node_labels_dic(path,'DD_node_labels.txt')
    node_dic2={}
    for k,v in node_dic.items():
        node_dic2[k]=v-1
    node_dic=node_dic2
    graphs=graph_label_list(path,'DD_graph_labels.txt')
    adjency=compute_adjency(path,'DD_A.txt')
    data_dict=graph_indicator(path,'DD_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],89)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data