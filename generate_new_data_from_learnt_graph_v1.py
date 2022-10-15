from logging import root
import numpy as np

import matplotlib.pyplot as plt

import os

import pandas as pd
import networkx as nx

from causal_graphs.graph_utils import adj_matrix_to_edges, edges_or_adj_matrix, sort_graph_by_vars, get_node_relations





#Relabel nodes
vars_mapping = {
    "$X_{1}$": "a0", "$X_{2}$": "a1", "$X_{3}$": "a2", "$X_{4}$": "a3", "$X_{5}$": "a4",
    "$X_{6}$": "b0", "$X_{7}$": "b1", "$X_{8}$": "b2", "$X_{9}$": "b3", "$X_{10}$": "b4",
    "$X_{11}$": "bo0", "$X_{12}$": "bo1", "$X_{13}$": "bo2",
    "$X_{14}$": "pt0", "$X_{15}$": "pt1", "$X_{16}$": "pt2", "$X_{17}$": "pt3",
    "$X_{18}$": "pl0", "$X_{19}$": "pl1", "$X_{20}$": "pl2",
    "$X_{21}$": "i0", "$X_{22}$": "i1", "$X_{23}$": "i2", "$X_{24}$": "i3", "$X_{25}$": "i4",
    "$X_{26}$": "pb0", "$X_{27}$": "pb1", "$X_{28}$": "pb2", "$X_{29}$": "pb3", "$X_{30}$": "pb4",
    "$X_{31}$": "c0", "$X_{32}$": "c1", "$X_{33}$": "c2", "$X_{34}$": "c3", "$X_{35}$": "c4",
    "$X_{36}$": "e0", "$X_{37}$": "e1", "$X_{38}$": "e2",
    # "$X_{39}$": "s0", "$X_{40}$": "s1", "$X_{41}$": "s2", "$X_{42}$": "s3" #Severity classes
    "$X_{39}$": "d0", "$X_{40}$": "d1", "$X_{41}$": "d2", "$X_{42}$": "d3", "$X_{43}$": "d4", "$X_{44}$": "d5", "$X_{45}$": "d6" #Disease classes
}


def drawGraph(Graph, reports_path = "."):


    labels = nx.get_edge_attributes(Graph, "weight")
    
    #Change float precision
    for k,v in labels.items():
        labels[k] = f'{v:0.2f}'

    A = nx.nx_agraph.to_agraph(Graph)        # convert to a graphviz graph
    A.layout(prog='dot')            # neato layout
    #A.draw('test3.pdf')

    root_nodes = np.unique([e1 for (e1, e2), v in labels.items()])
    root_nodes_colors = {}

    for idx, node in enumerate(root_nodes):
        color =  "#"+''.join([hex(np.random.randint(0,16))[-1] for i in range(6)])
        root_nodes_colors[node] = color

    for (e1, e2), v in labels.items():
        edge = A.get_edge(e1,e2)
        edge.attr['weight'] = v
        edge.attr['label'] = str(v)
        # edge.attr['color'] = "red:blue"
        edge.attr['color'] = root_nodes_colors[e1]
        
    A.draw(os.path.join(reports_path, f"adjacency_matrix_temp.png"),
            args='-Gnodesep=1.0 -Granksep=9.0 -Gfont_size=1', prog='dot' )  



def main():

    causal_discovery_exp_dir = "/home/grg/Research/ENCO/checkpoints/"
    
    acyclic_adj_matrix_path = "2022_23_Acyclic_DiseasePerPatient-max_GroundTruth_UpsampledTrainTest_sparity0.0004/binary_acyclic_matrix_001_DiseaseBiomarkersGTabTrainUpsampledPatient-max.npy"

    acyclic_adj_matrix = np.load(os.path.join(causal_discovery_exp_dir, acyclic_adj_matrix_path))

    print(f"acyclic_adj_matrix.shape = {acyclic_adj_matrix.shape}")

    adj_matrix = acyclic_adj_matrix

    ##Process adjacency matrix
    
    # num_classes = 4 #Severity-classes
    num_classes = 7 #Disease-classes

    REVERSE_OUTGOING_CLASS_EDGES = True

    if not REVERSE_OUTGOING_CLASS_EDGES:
        #Mask outgoing edges that start from severity to other features
        # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes
    else:
        #Reverse the edge direction of severity classes; convert outgoing edges to incoming edges 
        # for s_idx in [38, 39, 40, 41]: #Severity classes
        # for s_idx in [38, 39, 40, 41, 42, 43, 44]: #Disease classes
        for s_idx in range(38, 38+num_classes): #Disease classes
            outgoing_edges = adj_matrix[s_idx, :]
            outgoing_nodes = np.where(outgoing_edges == 1)
            adj_matrix[outgoing_nodes, s_idx] = 1

    # # TODO-GRG: Need to check if we need to transpose the adj_matrix or not!!!
    # # Transpose for mask because adj[i,j] means that i->j
    # mask_adj_matrices = adj_matrix.transpose(1, 2)

    # variables = list(range(45))
    variables = list(vars_mapping.values())

    G = pd.DataFrame(adj_matrix, index = variables, columns = variables)
    G = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)
    
    #Draw graph

    DRAW_GRAPH = False
    if DRAW_GRAPH:
        # nx.draw_networkx(G)
        # plt.savefig('Graph_temp.png')
        # plt.show()
        drawGraph(G)

    #Node degrees 
    node_degrees = {}
    for var in variables:
        node_degrees[var]= G.degree[var]

    #Node in-degrees 
    node_in_degrees = {}
    for var in variables:
        node_in_degrees[var]= G.in_degree[var]

    in_degree_sort_idx = np.argsort(list(node_in_degrees.values()))
    ascending_in_degree_nodes =  np.array(variables.copy())[in_degree_sort_idx]

    #Node out-degrees 
    node_out_degrees = {}
    for var in variables:
        node_out_degrees[var]= G.out_degree[var]

    out_degree_sort_idx = np.argsort(list(node_out_degrees.values()))
    ascending_out_degree_nodes =  np.array(variables.copy())[out_degree_sort_idx]

    #Node parents
    node_parents = {}
    for var in variables:
        node_parents[var]= list(G.predecessors(var))

    unconnected_nodes = [v for v,d in node_degrees.items() if d == 0]
    print(f"Unconnected nodes = {unconnected_nodes}")

    root_nodes = [n for n,d in G.in_degree() if d == 0 and G.degree(n) != 0] 
    print(f"Root nodes = {root_nodes}")

    # nx.ancestors(G, 'd0')

    sorted_variables, edges, adj_matrix, sorted_idxs = sort_graph_by_vars(variables, adj_matrix = adj_matrix)

    # sorted_variables = np.array(variables)[sorted_idxs]

    # paths_dict = {}
    # for node in G:
    #     if G.out_degree(node)==0: #it's a leaf
    #         paths_dict[node] = nx.shortest_path(G, root, node)

    for var in sorted_variables:
        print(f"{var}: {node_parents[var]}")

    pass



if __name__ == "__main__":
    print("Started...")
    main()
    print("Finished!")