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



# Runtime optimization of sklearn uisng Intel - Ref: https://github.com/intel/scikit-learn-intelex
from sklearnex import patch_sklearn, config_context
patch_sklearn()


def main():

    # task = 'heart-disease' #'heart-disease-binary' #'heart-disease' #'parity5' #'labor'
    # task = 'cifar-t1'
    # task = 'higgs_small-t1'
    task = 'credit_card_fraud-size3'

    
    ### Load training data
    
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
    

    # category_classes = list(range(38, 38+num_classes))
    category_classes = list(range(num_vars, num_vars+num_classes))



    print(f"train_data.shape = {train_data.shape}")

    train_features = train_data[:, :num_vars]

    train_labels = train_data[:, category_classes]

    # unique_train_data = np.unique(train_data, axis=0)
    unique_train_data = np.vstack({tuple(row) for row in train_data})
    print(f"unique_train_data = {unique_train_data.shape}")

    unique_train_labels = unique_train_data[:, category_classes]

    classes, count = np.unique(unique_train_labels, return_counts = True)
    print(f"[unique_train_data] classes, count = {classes, count}")   

    pass



if __name__ == "__main__":
    print("Started...")
    main()
    print("Finished!")