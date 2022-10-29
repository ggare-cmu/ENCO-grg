from nis import cat
import numpy as np

import os

import pandas as pd
import networkx as nx

from causal_graphs.graph_utils import adj_matrix_to_edges, edges_or_adj_matrix, sort_graph_by_vars, get_node_relations


import utils


from numpy.random import default_rng
rng = default_rng()



def main():

    exp_name = "randomGraphGenerator"
    
    # gt_task = 'credit_card_fraud-size4'
    # ref_task = 'credit_card_fraud-size3'
    # gt_task = 'higgs_small-size5'
    # ref_task = 'higgs_small-size1'
    # gt_task = 'heart-disease-binary-t3'
    # ref_task = 'heart-disease-binary-t5'

    gt_task = 'credit_card_fraud-size2'
    ref_task = 'credit_card_fraud-size2'

    # trial = 'randomGraph_R3'
    # trial = 'randomGraph_P30_R3' #Plus 30 edges
    trial = 'randomGraph_N30_R3' #Minus 30 edges


    # gt_causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_22_Acyclic_{gt_task}_TrainUpsampledPatient_T1"
    # ref_causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_22_Acyclic_{ref_task}_TrainUpsampledPatient_T1"
    # gt_causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_26_Acyclic_{gt_task}_TrainUpsampledPatient" #Heart-disease
    # ref_causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_26_Acyclic_{ref_task}_TrainUpsampledPatient" #Heart-disease
    gt_causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_22_Acyclic_{gt_task}_TrainUpsampledPatient_T1"
    ref_causal_discovery_exp_dir = f"/home/grg/Research/ENCO-grg/checkpoints/2022_10_22_Acyclic_{ref_task}_TrainUpsampledPatient_T1_{trial}"
    

    #Load adjacancy matrix

    gt_acyclic_adj_matrix = np.load(os.path.join(gt_causal_discovery_exp_dir, f"binary_acyclic_matrix_001_{gt_task}_TrainUpsampledPatient.npy"))
    ref_acyclic_adj_matrix = np.load(os.path.join(ref_causal_discovery_exp_dir, f"binary_acyclic_matrix_001_{ref_task}_TrainUpsampledPatient.npy"))


    




    diff = np.abs(1.0*gt_acyclic_adj_matrix - 1.0*ref_acyclic_adj_matrix)

    shd_double_for_anticausal = np.sum(diff)

    diff = diff + diff.transpose()
    diff[diff > 1] = 1  # Ignoring the double edges.
    shd = np.sum(diff)/2


    # print(f"Structural Hamming Dist between {gt_task} & {ref_task} double_for_anticausal = {shd_double_for_anticausal}")
    # print(f"Structural Hamming Dist between {gt_task} & {ref_task} = {shd}")
    print(f"Structural Hamming Dist between {gt_task} & {ref_task}_{trial} double_for_anticausal = {shd_double_for_anticausal}")
    print(f"Structural Hamming Dist between {gt_task} & {ref_task}_{trial} = {shd}")

    # #Code-Source: https://github.com/ElementAI/causal_discovery_toolbox/blob/master/cdt/metrics.py
    # if double_for_anticausal:
    #     return np.sum(diff)
    # else:
    #     diff = diff + diff.transpose()
    #     diff[diff > 1] = 1  # Ignoring the double edges.
    #     return np.sum(diff)/2


if __name__ == "__main__":
    print("Started...")
    main()
    print("Finished!")