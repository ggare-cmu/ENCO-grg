import torch
import numpy as np
import time
from argparse import ArgumentParser
from copy import deepcopy
import json
import random
import os
import sys
sys.path.append("../")

from causal_graphs.graph_utils import adj_matrix_to_edges
from causal_graphs.graph_visualization import visualize_graph, visualize_graph_with_relabeling
from causal_discovery.utils import get_device
from causal_discovery.enco import ENCO


import utils

def set_seed(seed):
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


def get_basic_parser():
    """
    Returns argument parser of standard hyperparameters/experiment arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs to run ENCO for.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the experiments.')
    parser.add_argument('--cluster', action='store_true',
                        help='If True, no tqdm progress bars are used.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size to use for distribution and graph fitting.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size of the distribution fitting NNs.')
    parser.add_argument('--use_flow_model', action='store_true',
                        help='If True, a Deep Sigmoidal Flow will be used as model if'
                             ' the graph contains continuous data.')
    parser.add_argument('--model_iters', type=int, default=1000,
                        help='Number of updates per distribution fitting stage.')
    parser.add_argument('--graph_iters', type=int, default=100,
                        help='Number of updates per graph fitting stage.')
    parser.add_argument('--lambda_sparse', type=float, default=0.004,
                        help='Sparsity regularizer in the graph fitting stage.')
    parser.add_argument('--lr_model', type=float, default=5e-3,
                        help='Learning rate of distribution fitting NNs.')
    parser.add_argument('--lr_gamma', type=float, default=2e-2,
                        help='Learning rate of gamma parameters in graph fitting.')
    parser.add_argument('--lr_theta', type=float, default=1e-1,
                        help='Learning rate of theta parameters in graph fitting.')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay to use during distribution fitting.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save experiment log to. If None, one will'
                             ' be created based on the current time')
    parser.add_argument('--GF_num_batches', type=int, default=1,
                        help='Number of batches to use in graph fitting gradient estimators.')
    parser.add_argument('--GF_num_graphs', type=int, default=100,
                        help='Number of graph samples to use in the gradient estimators.')
    parser.add_argument('--max_graph_stacking', type=int, default=200,
                        help='Number of graphs to evaluate in parallel. Reduce this to save memory.')
    parser.add_argument('--use_theta_only_stage', action='store_true',
                        help='If True, gamma is frozen in every second graph fitting stage.'
                             ' Recommended for large graphs with >=100 nodes.')
    parser.add_argument('--theta_only_num_graphs', type=int, default=4,
                        help='Number of graph samples to use when gamma is frozen.')
    parser.add_argument('--theta_only_iters', type=int, default=1000,
                        help='Number of updates per graph fitting stage when gamma is frozen.')
    parser.add_argument('--save_model', action='store_true',
                        help='If True, the neural networks will be saved besides gamma and theta.')
    parser.add_argument('--stop_early', action='store_true',
                        help='If True, ENCO stops running if it achieved perfect reconstruction in'
                             ' all of the last 5 epochs.')
    parser.add_argument('--sample_size_obs', type=int, default=5000,
                        help='Dataset size to use for observational data. If an exported graph is'
                             ' given as input and sample_size_obs is smaller than the exported'
                             ' observational dataset, the first sample_size_obs samples will be taken.')
    parser.add_argument('--sample_size_inters', type=int, default=200,
                        help='Number of samples to use per intervention. If an exported graph is'
                             ' given as input and sample_size_inters is smaller than the exported'
                             ' interventional dataset, the first sample_size_inters samples will be taken.')
    parser.add_argument('--max_inters', type=int, default=-1,
                        help='Number of variables to provide interventional data for. If smaller'
                             ' than zero, interventions on all variables will be used.')
    return parser


def test_graph(graph, args, checkpoint_dir, file_id):
    """
    Runs ENCO on a given graph for structure learning.

    Parameters
    ----------
    graph : CausalDAG
            The graph on which we want to perform causal structure learning.
    args : Namespace
           Parsed input arguments from the argument parser, including all
           hyperparameters.
    checkpoint_dir : str
                     Directory to which all logs and the model should be
                     saved to.
    file_id : str
              Identifier of the graph/experiment instance. Is used for creating
              log filenames, and identify the graph among other experiments in
              the same checkpoint directory.
    """
    # Determine variables to exclude from the intervention set
    if args.max_inters < 0:
        graph.exclude_inters = None
    elif args.max_inters == 0: #GRG: To exclude intervention on all variables
        graph.exclude_inters = list(range(graph.num_vars))
    elif graph.exclude_inters is not None:
        graph.exclude_inters = graph.exclude_inters[:-args.max_inters]
    else:
        exclude_inters = list(range(graph.num_vars))
        random.seed(args.seed)
        random.shuffle(exclude_inters)
        exclude_inters = exclude_inters[:-args.max_inters]
        graph.exclude_inters = exclude_inters

    # Execute ENCO on graph
    discovery_module = ENCO(graph=graph,
                            hidden_dims=[args.hidden_size],
                            use_flow_model=args.use_flow_model,
                            lr_model=args.lr_model,
                            weight_decay=args.weight_decay,
                            lr_gamma=args.lr_gamma,
                            lr_theta=args.lr_theta,
                            model_iters=args.model_iters,
                            graph_iters=args.graph_iters,
                            batch_size=args.batch_size,
                            GF_num_batches=args.GF_num_batches,
                            GF_num_graphs=args.GF_num_graphs,
                            lambda_sparse=args.lambda_sparse,
                            use_theta_only_stage=args.use_theta_only_stage,
                            theta_only_num_graphs=args.theta_only_num_graphs,
                            theta_only_iters=args.theta_only_iters,
                            max_graph_stacking=args.max_graph_stacking,
                            sample_size_obs=args.sample_size_obs,
                            sample_size_inters=args.sample_size_inters
                            )
    discovery_module.to(get_device())
    start_time = time.time()
    discovery_module.discover_graph(num_epochs=args.num_epochs,
                                    stop_early=args.stop_early)
    duration = int(time.time() - start_time)
    print("-> Finished training in %ih %imin %is" % (duration // 3600, (duration // 60) % 60, duration % 60))

    # Save metrics in checkpoint folder
    metrics = discovery_module.get_metrics()
    with open(os.path.join(checkpoint_dir, "metrics_%s.json" % file_id), "w") as f:
        json.dump(metrics, f, indent=4)
    print('-'*50 + '\nFinal metrics:')
    discovery_module.print_graph_statistics(m=metrics)
    if graph.num_vars < 100:
        metrics_acyclic = discovery_module.get_metrics(enforce_acyclic_graph=True)
        with open(os.path.join(checkpoint_dir, "metrics_acyclic_%s.json" % file_id), "w") as f:
            json.dump(metrics_acyclic, f, indent=4)
        print('-'*50 + '\nFinal metrics (acyclic):')
        discovery_module.print_graph_statistics(m=metrics_acyclic)
    with open(os.path.join(checkpoint_dir, "metrics_full_log_%s.json" % file_id), "w") as f:
        json.dump(discovery_module.metric_log, f, indent=4)

    # Save predicted binary matrix
    binary_matrix = discovery_module.get_binary_adjmatrix().detach().cpu().numpy()
    np.save(os.path.join(checkpoint_dir, 'binary_matrix_%s.npy' % file_id),
            binary_matrix.astype(np.bool))
    if graph.num_vars < 100:
        acyclic_matrix = discovery_module.get_acyclic_adjmatrix().detach().numpy()
        np.save(os.path.join(checkpoint_dir, 'binary_acyclic_matrix_%s.npy' % file_id),
                acyclic_matrix.astype(np.bool))

    # Visualize predicted graphs. For large graphs, visualizing them do not really help
    # if graph.num_vars < 40:
    if graph.num_vars < 50:
        pred_graph = deepcopy(graph)
        # pred_graph.adj_matrix = binary_matrix
        pred_graph.adj_matrix = acyclic_matrix
        pred_graph.edges = adj_matrix_to_edges(pred_graph.adj_matrix)
        figsize = max(3, pred_graph.num_vars / 1.5)


        # visualize_graph(pred_graph,
        #                 filename=os.path.join(checkpoint_dir, "graph_%s_prediction.pdf" % (file_id)),
        #                 figsize=(figsize, figsize),
        #                 layout="circular")


        #Relabel nodes
        mapping = {
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

        pred_graph = visualize_graph_with_relabeling(pred_graph, mapping,
                        filename=os.path.join(checkpoint_dir, "graph_%s_prediction.pdf" % (file_id)),
                        figsize=(figsize, figsize),
                        layout="circular")
        
        logger = utils.Logger(filename = os.path.join(checkpoint_dir, "graph_%s_logs.txt" % (file_id)))

        logger.log(f"# of Edges = {len(pred_graph.edges)}")
        logger.log(f"Edges = {pred_graph.edges}")

        # s0_edges = [i for i in pred_graph.edges if 's0' in str(i)]
        # s1_edges = [i for i in pred_graph.edges if 's1' in str(i)]
        # s2_edges = [i for i in pred_graph.edges if 's2' in str(i)]
        # s3_edges = [i for i in pred_graph.edges if 's3' in str(i)]
        
        # severity_correlated_nodes = []

        # for a, b in s0_edges + s1_edges + s2_edges + s3_edges:

        #     if a != 's0' and a != 's1' and a != 's2' and a != 's3':
        #         severity_correlated_nodes.append(a)
        #     elif b != 's0' and b != 's1' and b != 's2' and b != 's3':
        #         severity_correlated_nodes.append(b)
        #     else:
        #         raise Exception(f"Error! Detected edge between severity {(a,b)}")

        # severity_correlated_nodes = np.unique(severity_correlated_nodes).tolist()

        # logger.log(f"severity-0 edges = {s0_edges}")
        # logger.log(f"severity-1 edges = {s1_edges}")
        # logger.log(f"severity-2 edges = {s2_edges}")
        # logger.log(f"severity-3 edges = {s3_edges}")

        # logger.log(f"Found {len(severity_correlated_nodes)} severity_correlated_nodes = {severity_correlated_nodes}")

        d0_edges = [i for i in pred_graph.edges if 'd0' in str(i)]
        d1_edges = [i for i in pred_graph.edges if 'd1' in str(i)]
        d2_edges = [i for i in pred_graph.edges if 'd2' in str(i)]
        d3_edges = [i for i in pred_graph.edges if 'd3' in str(i)]
        d4_edges = [i for i in pred_graph.edges if 'd4' in str(i)]
        d5_edges = [i for i in pred_graph.edges if 'd5' in str(i)]
        d6_edges = [i for i in pred_graph.edges if 'd6' in str(i)]
        
        disease_correlated_nodes = []

        for a, b in d0_edges + d1_edges + d2_edges + d3_edges + d4_edges + d5_edges + d6_edges:

            if a != 'd0' and a != 'd1' and a != 'd2' and a != 'd3' and a != 'd4' and a != 'd5' and a != 'd6':
                disease_correlated_nodes.append(a)
            elif b != 'd0' and b != 'd1' and b != 'd2' and b != 'd3' and b != 'd4' and b != 'd5' and b != 'd6':
                disease_correlated_nodes.append(b)
            else:
                print(f"Error! Detected edge between severity {(a,b)}")
                # raise Exception(f"Error! Detected edge between severity {(a,b)}")

        disease_correlated_nodes = np.unique(disease_correlated_nodes).tolist()

        logger.log(f"disease-0 edges = {d0_edges}")
        logger.log(f"disease-1 edges = {d1_edges}")
        logger.log(f"disease-2 edges = {d2_edges}")
        logger.log(f"disease-3 edges = {d3_edges}")
        logger.log(f"disease-4 edges = {d4_edges}")
        logger.log(f"disease-5 edges = {d5_edges}")
        logger.log(f"disease-6 edges = {d6_edges}")

        logger.log(f"Found {len(disease_correlated_nodes)} disease_correlated_nodes = {disease_correlated_nodes}")

        logger.close()

    

    # Save parameters and model if wanted
    state_dict = discovery_module.get_state_dict()
    if not args.save_model:  # The model can be expensive in memory
        _ = state_dict.pop("model")
    torch.save(state_dict,
               os.path.join(checkpoint_dir, "state_dict_%s.tar" % file_id))


def visualizeGraph(graph, adj_matrix, filename):


    # Visualize predicted graphs. For large graphs, visualizing them do not really help
    # if graph.num_vars < 40:
    if graph.num_vars < 50:
        pred_graph = deepcopy(graph)
        pred_graph.adj_matrix = adj_matrix
        pred_graph.edges = adj_matrix_to_edges(pred_graph.adj_matrix)
        figsize = max(3, pred_graph.num_vars / 1.5)


        # visualize_graph(pred_graph,
        #                 filename=os.path.join(checkpoint_dir, "graph_%s_prediction.pdf" % (file_id)),
        #                 figsize=(figsize, figsize),
        #                 layout="circular")


        #Relabel nodes
        mapping = {
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

        pred_graph = visualize_graph_with_relabeling(pred_graph, mapping,
                        filename=filename,
                        figsize=(figsize, figsize),
                        layout="circular")
        
        logger = utils.Logger(filename = filename.replace('.pdf', '.txt'))

        logger.log(f"# of Edges = {len(pred_graph.edges)}")
        logger.log(f"Edges = {pred_graph.edges}")

        logger.close()

    


import torch.utils.data as data
from causal_discovery.datasets import ObservationalCategoricalData

def pred_graph(graph, args, checkpoint_dir, file_id):
    """
    Runs ENCO on a given graph for prediction.

    Parameters
    ----------
    graph : CausalDAG
            The graph on which we want to perform causal structure learning.
    args : Namespace
           Parsed input arguments from the argument parser, including all
           hyperparameters.
    checkpoint_dir : str
                     Directory to which all logs and the model should be
                     saved to.
    file_id : str
              Identifier of the graph/experiment instance. Is used for creating
              log filenames, and identify the graph among other experiments in
              the same checkpoint directory.
    """
    # Determine variables to exclude from the intervention set
    if args.max_inters < 0:
        graph.exclude_inters = None
    elif graph.exclude_inters is not None:
        graph.exclude_inters = graph.exclude_inters[:-args.max_inters]
    else:
        exclude_inters = list(range(graph.num_vars))
        random.seed(args.seed)
        random.shuffle(exclude_inters)
        exclude_inters = exclude_inters[:-args.max_inters]
        graph.exclude_inters = exclude_inters

    # Execute ENCO on graph
    discovery_module = ENCO(graph=graph,
                            hidden_dims=[args.hidden_size],
                            use_flow_model=args.use_flow_model,
                            lr_model=args.lr_model,
                            weight_decay=args.weight_decay,
                            lr_gamma=args.lr_gamma,
                            lr_theta=args.lr_theta,
                            model_iters=args.model_iters,
                            graph_iters=args.graph_iters,
                            batch_size=args.batch_size,
                            GF_num_batches=args.GF_num_batches,
                            GF_num_graphs=args.GF_num_graphs,
                            lambda_sparse=args.lambda_sparse,
                            use_theta_only_stage=args.use_theta_only_stage,
                            theta_only_num_graphs=args.theta_only_num_graphs,
                            theta_only_iters=args.theta_only_iters,
                            max_graph_stacking=args.max_graph_stacking,
                            sample_size_obs=args.sample_size_obs,
                            sample_size_inters=args.sample_size_inters
                            )
    
    #Load checkpoint
    state_dict = torch.load(os.path.join(checkpoint_dir, f"state_dict_{file_id.replace('Test', 'Train')}.tar"))
    discovery_module.load_state_dict(state_dict)
    
    # Create observational dataset
    test_obs_dataset = ObservationalCategoricalData(graph, dataset_size=args.sample_size_obs)
    test_obs_data_loader = data.DataLoader(test_obs_dataset, batch_size=args.batch_size,
                                        shuffle=False, drop_last=False)
                                        
    adj_matrices = np.load(os.path.join(checkpoint_dir, f"binary_acyclic_matrix_{file_id.replace('Test', 'Train')}.npy")) #ACyclic Graph
    # adj_matrices = np.load(os.path.join(checkpoint_dir, f"binary_matrix_{file_id.replace('Test', 'Train')}.npy")) #Cyclic Graph

    original_graph_path = os.path.join(checkpoint_dir, "graph_%s_original_prediction.pdf" % (file_id))
    visualizeGraph(graph, adj_matrices, original_graph_path)

    '''
    adj_matrices : torch.FloatTensor, shape [num_vars, num_vars]
                    Float tensor with values between 0 and 1. An element (i,j)
                    represents the probability of having an edge from X_i to X_j,
                    i.e., not masking input X_i for predicting X_j.
    '''

    # num_classes = 4 #Severity-classes
    num_classes = 7 #Disease-classes

    REVERSE_OUTGOING_CLASS_EDGES = True

    if not REVERSE_OUTGOING_CLASS_EDGES:
        #Mask outgoing edges that start from severity to other features
        # adj_matrices[-4:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Severity classes
        adj_matrices[-num_classes:, :] = 0 #Rows represent outgoing edges and Columns represent incoming edges #Disease classes
    else:
        #Reverse the edge direction of severity classes; convert outgoing edges to incoming edges 
        # for s_idx in [38, 39, 40, 41]: #Severity classes
        # for s_idx in [38, 39, 40, 41, 42, 43, 44]: #Disease classes
        for s_idx in range(38, 38+num_classes): #Disease classes
            outgoing_edges = adj_matrices[s_idx, :]
            outgoing_nodes = np.where(outgoing_edges == 1)
            adj_matrices[outgoing_nodes, s_idx] = 1

    masked_graph_path = os.path.join(checkpoint_dir, "graph_%s_masked_prediction.pdf" % (file_id))
    visualizeGraph(graph, adj_matrices, masked_graph_path)

    logger = utils.Logger(filename = os.path.join(checkpoint_dir, "Testset_predictions_%s_results.txt" % (file_id)))


    discovery_module.to(get_device())
    start_time = time.time()
    # discovery_module.discover_graph(num_epochs=args.num_epochs,
    #                                 stop_early=args.stop_early)

    discovery_module.predict_using_graph(test_obs_data_loader, adj_matrices, num_classes, logger)
    duration = int(time.time() - start_time)
    logger.log("-> Finished prediction in %ih %imin %is" % (duration // 3600, (duration // 60) % 60, duration % 60))


    logger.close()
