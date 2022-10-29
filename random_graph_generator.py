from nis import cat
import numpy as np

import os

import pandas as pd
import networkx as nx

from causal_graphs.graph_utils import adj_matrix_to_edges, edges_or_adj_matrix, sort_graph_by_vars, get_node_relations


import utils


from numpy.random import default_rng
rng = default_rng()




def drawGraph(Graph, reports_path = ".", ):


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


#Code-Source: https://stackoverflow.com/questions/68496631/fastest-way-to-convert-an-undirected-cyclic-graph-ucg-to-a-directed-acyclic-gr
def ucg2dag(adj_matrix, starts):
    adj_list = [
        [target for target, is_connected in enumerate(row) if is_connected]
            for row in adj_matrix
    ]

    frontier = starts

    dag = [[] for _ in range(len(adj_list))]

    while frontier:
        for source in frontier:
            dag[source].extend(target for target in adj_list[source] if not target in starts)
        frontier = set(target 
            for source in frontier for target in adj_list[source] if not target in starts
        )
        starts.update(frontier)

    return dag





def generate_dag(nodes, parents_max, verbose = True):
    """ Create the structure of the graph """

    adjacency_matrix = np.zeros((nodes, nodes))

    for j in range(1, nodes):
        nb_parents = np.random.randint(0, min([parents_max, j])+1)
        # nb_parents = np.random.randint(0, j+1)
        for i in np.random.choice(range(0, j), nb_parents, replace=False):
            adjacency_matrix[i, j] = 1

    try:
        g = nx.DiGraph(adjacency_matrix)
        assert not list(nx.simple_cycles(g))

    except AssertionError:
        if verbose:
            print("Regenerating, graph non valid...")
        generate_dag()

    original_adjacency_matrix = np.copy(adjacency_matrix)

    return adjacency_matrix


def main():

    exp_name = "randomGraphGenerator"
    
    # task = 'heart-disease' #'heart-disease-binary' #'heart-disease' #'parity5' #'labor'
    # task = 'cifar-t1'
    # task = 'higgs_small-t1'
    task = 'credit_card_fraud-size2'

    # trial = 'randomGraph_R1'
    # trial = 'randomGraph_P30_R1' #Plus 30 edges
    trial = 'randomGraph_N30_R2' #Minus 30 edges

    causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_22_Acyclic_{task}_TrainUpsampledPatient_T1"
    
    
    DRAW_GRAPH = True


    print(f"Generating random graph for task {task} trial {trial}")


    #Create results dir
    results_dir = os.path.join(f"{causal_discovery_exp_dir}_{trial}")
    utils.createDirIfDoesntExists(results_dir)


    #Load adjacancy matrix

    acyclic_adj_matrix_path = f"binary_acyclic_matrix_001_{task}_TrainUpsampledPatient.npy"

    acyclic_adj_matrix = np.load(os.path.join(causal_discovery_exp_dir, acyclic_adj_matrix_path))

    print(f"acyclic_adj_matrix.shape = {acyclic_adj_matrix.shape}")

    org_adj_matrix = acyclic_adj_matrix 

    num_edges = org_adj_matrix.sum()

    #Chanmge is more or less edges needed
    # num_edges = num_edges + 30 #Plus 30 edges
    num_edges = num_edges - 30 #Minus 30 edges
    
    parents_max = 5 #For 80 edges
    # parents_max = 6 #For 110 edges

    print(f"Required num of edges = {num_edges}")



    ### Load training data
    
    train_data_path = f"datasets/{task}_TrainUpsampledPatient.npz"

    train_dataset = np.load(train_data_path, allow_pickle = True)

    vars_list = train_dataset['vars_list']
    class_list = train_dataset['class_list']

    num_vars = len(vars_list)
    num_classes = len(class_list)

    features_list = vars_list.tolist() + class_list.tolist()

    variables = features_list


    
    #Generate random graph 
    Regenerate = True

    while Regenerate:
        
        Regenerate = False
    
        rand_num_edges = -1

        while rand_num_edges != num_edges:

            # rand_adj_matrix = np.zeros_like(adj_matrix)
            
            # x0_idx = rng.choice(np.arange(adj_matrix.shape[0]), size = num_edges, replace = True)
            # x1_idx = rng.choice(np.arange(adj_matrix.shape[1]), size = num_edges, replace = True)

            # # x_direction = rng.binomial(n = 1, p = 0.5, size = num_edges)

            # rand_adj_matrix[x0_idx, x1_idx] = 1

            # rand_adj_matrix = nx.gnp_random_graph(adj_matrix.shape[0], num_edges/(adj_matrix.shape[0]*adj_matrix.shape[1]), directed=False)

            # num_starts = rng.choice(np.arange(adj_matrix.shape[0]), size = 1)
            # starts = rng.choice(np.arange(adj_matrix.shape[0]), size = num_starts, replace = False)
            
            # rand_adj_matrix = ucg2dag(rand_adj_matrix, starts)

            # if not nx.is_directed_acyclic_graph(rand_adj_matrix):
            #     continue
            
            rand_adj_matrix = generate_dag(org_adj_matrix.shape[0], parents_max)
            
            if not nx.is_directed_acyclic_graph(nx.DiGraph(rand_adj_matrix)):
                continue

            rand_num_edges = rand_adj_matrix.sum()
            print(rand_num_edges)

        # assert rand_adj_matrix.sum() == org_adj_matrix.sum(), "Error! Num of edges not same."
        assert rand_num_edges == num_edges, "Error! Num of edges not same."


        ##Process adjacency matrix   

        REVERSE_OUTGOING_CLASS_EDGES = True

        if not REVERSE_OUTGOING_CLASS_EDGES:
            #Mask outgoing edges that start from severity to other features
            # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
            rand_adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes
        else:
            #Reverse the edge direction of severity classes; convert outgoing edges to incoming edges 
            # for s_idx in [38, 39, 40, 41]: #Severity classes
            # for s_idx in [38, 39, 40, 41, 42, 43, 44]: #Disease classes
            for s_idx in range(num_vars, num_vars+num_classes): #Disease classes
                outgoing_edges = rand_adj_matrix[s_idx, :]
                outgoing_nodes = np.where(outgoing_edges == 1)
                rand_adj_matrix[outgoing_nodes, s_idx] = 1
            
            #TODO-GRG: Check this
            #Now remove outgoing edges - otherwise this introduces cycles
            
            #Mask outgoing edges that start from severity to other features
            # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
            rand_adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes

        #Check if acyclic 

        #Get sampling order
        try:
            sorted_variables, edges, sorted_rand_adj_matrix, sorted_idxs = sort_graph_by_vars(variables, adj_matrix = rand_adj_matrix)
        except:
            Regenerate = True


    print(f"Successfully generated a random acyclic graph that matches all the conditions!")

    # DRAW_GRAPH = False
    if DRAW_GRAPH:

        G = pd.DataFrame(rand_adj_matrix, index = variables, columns = variables)
        G = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)
        
        # nx.draw_networkx(G)
        # plt.savefig('Graph_temp.png')
        # plt.show()
        drawGraph(G, results_dir)

    #Save random graph
    np.save(os.path.join(results_dir, acyclic_adj_matrix_path), rand_adj_matrix)


if __name__ == "__main__":
    print("Started...")
    main()
    print("Finished!")