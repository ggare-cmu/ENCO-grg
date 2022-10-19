from ctypes import util
import numpy as np

import matplotlib.pyplot as plt

import torch

import os

import pandas as pd
import networkx as nx

from causal_graphs.graph_utils import adj_matrix_to_edges, edges_or_adj_matrix, sort_graph_by_vars, get_node_relations

import utils


import random



def set_seed(seed = 42):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



#Relabel nodes
vars_mapping = {
    "x1": "age", "x2": "sex",
    "x3": "cp", "x4": "trestbps",
    "x5": "chol", "x6": "fbs",
    "x7": "restecg", "x8": "thalach",
    "x9": "exang", "x10": "oldpeak",
    "x11": "slope", "x12": "ca",
    "x13": "thal", 
    "c1": "healthy", "c2": "heart-disease" #Disease classes
}


# vars_color_mapping = {
#     "x1": ["age", "#00A36C"], "x2": ["sex", "#FFC300"],
#     "x3": ["cp", "#FFC300"], "x4": ["trestbps", "#00A36C"],
#     "x5": ["chol", "#FFC300"], "x6": ["fbs", "#FFC300"],
#     "x7": ["restecg", "#FFC300"], "x8": ["thalach", "#00A36C"],
#     "x9": ["exang", "#FFC300"], "x10": ["oldpeak", "#FFC300"],
#     "x11": ["slope", "#00A36C"], "x12": ["ca", "#FFC300"],
#     "x13": ["thal", "#00A36C"], 
#     "c1": ["healthy", "#0096FF"], "c2": ["heart-disease", "#0096FF"] #Disease classes
# }


# vars_color_mapping = {
#     "age": "#00A36C", "sex": "#FFC300",
#     "cp": "#FFC300", "trestbps": "#00A36C",
#     "chol": "#FFC300", "fbs": "#FFC300",
#     "restecg": "#FFC300", "thalach": "#00A36C",
#     "exang": "#FFC300", "oldpeak": "#FFC300",
#     "slope": "#00A36C", "ca": "#FFC300",
#     "thal": "#00A36C", 
#     "healthy": "#0096FF", "heart-disease": "#0096FF" #Disease classes
# }




# vars_color_mapping = {
#     "age": "#00A36C", "sex": "#FFC300",
#     "cp": "#FFC300", "trestbps": "#00A36C",
#     "chol": "#FFC300", "fbs": "#FFC300",
#     "restecg": "#FFC300", "thalach": "#00A36C",
#     "exang": "#FFC300", "oldpeak": "#FFC300",
#     "slope": "#00A36C", "ca": "#FFC300",
#     "thal": "#00A36C", 
#     "healthy": "#0096FF", "heart-disease": "#0096FF" #Disease classes
# }

# class_name_dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}




vars_color_mapping = {
    "x1": "#00A36C", "x2": "#FFC300",
    "x4": "#FFC300", "x3": "#00A36C",
    "x5": "#FFC300", "x6": "#FFC300",
    "x7": "#FFC300", "x8": "#00A36C",
    "x9": "#FFC300", "x10": "#FFC300",
    "x11": "#00A36C", "x12": "#FFC300",
    "x13": "#00A36C", "x14": "#00A36C",
    "x15": "#FFC300", "x10": "#FFC300",
    "x11": "#00A36C", "x12": "#FFC300",
    "airplane": "#0096FF", "automobile": "#0096FF", #Disease classes
    "bird": "#0096FF", "cat": "#0096FF", #Disease classes
    "deer": "#0096FF", "dog": "#0096FF", #Disease classes
    "frog": "#0096FF", "horse": "#0096FF", #Disease classes
    "ship": "#0096FF", "truck": "#0096FF", #Disease classes
}

def drawGraph(Graph, reports_path = ".", ):

    # #Assign node colors
    # node_color_map = []
    # for node in Graph:
    #     node_color_map.append(vars_color_mapping[node][1])

    # #Relabel node names
    # Graph = nx.relabel_nodes(Graph, vars_mapping)

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


    #Assign node colors
    A.node_attr['style']='filled'
    for idx, node_name in enumerate(root_nodes):
        node = A.get_node(node_name)
        node.attr['color'] = vars_color_mapping[node_name]
        node.attr['fillcolor'] = vars_color_mapping[node_name]


    for (e1, e2), v in labels.items():
        edge = A.get_edge(e1,e2)
        # edge.attr['weight'] = v

        # edge.attr['label'] = str(v)
        # edge.attr['label'] = label
        
        # edge.attr['color'] = "red:blue"
        edge.attr['color'] = root_nodes_colors[e1]
        
    # A.draw(os.path.join(reports_path, f"adjacency_matrix_temp.png"),
    #         args='-Gnodesep=1.0 -Granksep=9.0 -Gfont_size=1', prog='dot' )  

    A.draw(os.path.join(reports_path, f"adjacency_matrix_temp.png"),
            args='-Gnodesep=0.7 -Granksep=1.0 -Gfont_size=1.5', prog='dot')  


def getSamplingOrder(adj_matrix, variables, logger, results_dir, DRAW_GRAPH = False):


    G = pd.DataFrame(adj_matrix, index = variables, columns = variables)
    G = nx.from_pandas_adjacency(G, create_using=nx.DiGraph)
    
    #Draw graph

    # DRAW_GRAPH = False
    if DRAW_GRAPH:
        # nx.draw_networkx(G)
        # plt.savefig('Graph_temp.png')
        # plt.show()
        drawGraph(G, results_dir)

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
    logger.log(f"Unconnected nodes = {unconnected_nodes}")

    root_nodes = [n for n,d in G.in_degree() if d == 0 and G.degree(n) != 0] 
    logger.log(f"Root nodes = {root_nodes}")

    # nx.ancestors(G, 'd0')

    sorted_variables, edges, adj_matrix, sorted_idxs = sort_graph_by_vars(variables, adj_matrix = adj_matrix)

    # sorted_variables = np.array(variables)[sorted_idxs]

    # paths_dict = {}
    # for node in G:
    #     if G.out_degree(node)==0: #it's a leaf
    #         paths_dict[node] = nx.shortest_path(G, root, node)

    for var in sorted_variables:
        logger.log(f"{var}: {node_parents[var]}")

    return G, sorted_variables, node_parents, root_nodes, unconnected_nodes




def upsampleFeatures(labels, features):

    classes, count = np.unique(labels, return_counts = True)
    print(f"[Pre-Upsampling] classes, count = {classes, count}")   
    
    max_count = max(count)

    label_indices = []
    for c in classes:

        c_idx = np.where(labels == c)[0]
        assert np.unique(labels[c_idx]) == c, "Error! Wrong class index filtered."

        #Bug-GRG : Since we sample randomly some of the videos are never sampled/included. 
        # So, make sure to only sample additional required videos after including all videos at least once!
        #For the max count class, set replace to False as setting it True might exclude some samples from training
        # upsample_c_idx = np.random.choice(c_idx, size = max_count, replace = len(c_idx) < max_count)
        if len(c_idx) < max_count:
            # upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = len(c_idx) < max_count).tolist())
            upsample_c_idx = np.array(c_idx.tolist() + np.random.choice(c_idx, size = max_count - len(c_idx), replace = max_count > 2*len(c_idx)).tolist())
        else:
            upsample_c_idx = c_idx
        
        np.random.shuffle(upsample_c_idx)
        
        assert c_idx.shape == np.unique(upsample_c_idx).shape, "Error! Some videos where excluded on updampling."

        label_indices.extend(upsample_c_idx)

    assert len(label_indices) == max_count * len(classes)

    upsample_label_indices = label_indices

    # upsampled_features = features[label_indices, :]
    upsampled_features = features[label_indices]

    upsampled_labels = labels[label_indices]

    classes, count = np.unique(upsampled_labels, return_counts = True)
    print(f"[Post-Upsampling] classes, count = {classes, count}")   

    assert np.array_equal(count, max_count * np.ones(len(classes))), "Error! Upsampling didn't result in class-balance"

    return upsampled_labels, upsampled_features, upsample_label_indices



def main():

    exp_name = "generateContinousSamples"
    
    # task = 'diabetes-t5' #'heart-disease-binary-t2' #'dermatology' #'heart-disease-binary'
    task = 'cifar-t1'

    # causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_26_Acyclic_{task}_TrainUpsampledPatient (draw)"
    # causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_4_Acyclic_{task}_TrainUpsampledPatient"
    causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_4_Acyclic_{task}_TrainUpsampledPatient_T1"

    # set_seed()

    num_samples = 50 #200 #3000 #500 #1000
    # Binary_Features = False #True
    # features_type = 'continous' #categorical, binary, continous

    ADD_NOISE = False #False

    DRAW_GRAPH = True #False
    
    causal_mlp_hidden_layer_sizes = (128, 64, 32) #(32, 16) #(128, 64, 32)
    max_iter = 200 #200

    results_dir = os.path.join(causal_discovery_exp_dir, f"{exp_name}_{num_samples}")
    utils.createDirIfDoesntExists(results_dir)

    acyclic_adj_matrix_path = f"binary_acyclic_matrix_001_{task}_TrainUpsampledPatient.npy"

    acyclic_adj_matrix = np.load(os.path.join(causal_discovery_exp_dir, acyclic_adj_matrix_path))

    print(f"acyclic_adj_matrix.shape = {acyclic_adj_matrix.shape}")

    adj_matrix = acyclic_adj_matrix 
        
    ### Load training data
    
    # train_data_path = "DiseaseBiomarkersGTabTrainUpsampledPatient-max.npz"
    # train_data_path = "DiseaseModelBiomarkersGTabTrainUpsampledPatient-max.npz"
    # train_data_path = "labor_TrainUpsampledPatient.npz"
    train_data_path = f"datasets/{task}_TrainUpsampledPatient.npz"

    train_dataset = np.load(train_data_path, allow_pickle = True)

    train_data = train_dataset['data_obs']


    vars_list = train_dataset['vars_list']
    class_list = train_dataset['class_list']

    feature_type_dict = train_dataset['feature_type'].item()

    features_list = vars_list.tolist() + class_list.tolist()
    



    node_names_mapping = {}
    for idx, var in enumerate(features_list):
        # node_names_mapping[f"$X_\\{{idx}\\}$"] = var
        node_names_mapping["$X_{" + str(idx+1) + "}$"] = var

    causal_feature_mapping = {}
    for idx, var in enumerate(features_list):
        causal_feature_mapping[var] = idx

    num_vars = len(vars_list)
    num_classes = len(class_list)

    binary_cross_entropy = num_classes == 2

    num_categs = 1 + train_data.max(axis=0) #Bug-Fix GRG: Max should be taken along num_samples axis not num_vars axis
    assert len(num_categs) == adj_matrix.shape[0], "Error! Num categories does not match num of variables."
    new_categs_func = lambda i : num_categs[i]


    # # num_classes = 4 #Severity-classes
    # num_classes = 7 #Disease-classes

    # class_names = [ 'normal', 'covid', 'interstetial', 'copd asthma', 'chf', 'other-lung', 'others', ]
    class_names = class_list


    # category_classes = list(range(38, 38+num_classes))
    category_classes = list(range(num_vars, num_vars+num_classes))



    #Calculate Scores 
    reports_path = "."
    exp_name = "Using Synthetic Data"
    report_path = os.path.join(results_dir, f"classification_report_{task}.txt")
    logger = utils.Logger(report_path)

    logger.log(f"Classification report")

    logger.log(f"Exp name: {exp_name}")



    ##Process adjacency matrix   

    REVERSE_OUTGOING_CLASS_EDGES = True

    if not REVERSE_OUTGOING_CLASS_EDGES:
        #Mask outgoing edges that start from severity to other features
        # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes
    else:
        #Reverse the edge direction of severity classes; convert outgoing edges to incoming edges 
        # for s_idx in [38, 39, 40, 41]: #Severity classes
        # for s_idx in [38, 39, 40, 41, 42, 43, 44]: #Disease classes
        for s_idx in range(num_vars, num_vars+num_classes): #Disease classes
            outgoing_edges = adj_matrix[s_idx, :]
            outgoing_nodes = np.where(outgoing_edges == 1)
            adj_matrix[outgoing_nodes, s_idx] = 1
        
        #TODO-GRG: Check this
        #Now remove outgoing edges - otherwise this introduces cycles
        
        #Mask outgoing edges that start from severity to other features
        # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
        adj_matrix[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes

    # # TODO-GRG: Need to check if we need to transpose the adj_matrix or not!!!
    # # Transpose for mask because adj[i,j] means that i->j
    # mask_adj_matrices = adj_matrix.transpose(1, 2)

    # variables = list(range(45))
    # variables = list(vars_mapping.values())
    variables = features_list

    #Get sampling order
    G, sorted_variables, node_parents, root_nodes, unconnected_nodes = getSamplingOrder(adj_matrix, variables, logger, results_dir, DRAW_GRAPH)


    logger.close()

    pass



if __name__ == "__main__":
    print("Started...")
    main()
    print("Finished!")