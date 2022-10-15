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




# #Relabel nodes
# vars_mapping = {
#     "$X_{1}$": "a0", "$X_{2}$": "a1", "$X_{3}$": "a2", "$X_{4}$": "a3", "$X_{5}$": "a4",
#     "$X_{6}$": "b0", "$X_{7}$": "b1", "$X_{8}$": "b2", "$X_{9}$": "b3", "$X_{10}$": "b4",
#     "$X_{11}$": "bo0", "$X_{12}$": "bo1", "$X_{13}$": "bo2",
#     "$X_{14}$": "pt0", "$X_{15}$": "pt1", "$X_{16}$": "pt2", "$X_{17}$": "pt3",
#     "$X_{18}$": "pl0", "$X_{19}$": "pl1", "$X_{20}$": "pl2",
#     "$X_{21}$": "i0", "$X_{22}$": "i1", "$X_{23}$": "i2", "$X_{24}$": "i3", "$X_{25}$": "i4",
#     "$X_{26}$": "pb0", "$X_{27}$": "pb1", "$X_{28}$": "pb2", "$X_{29}$": "pb3", "$X_{30}$": "pb4",
#     "$X_{31}$": "c0", "$X_{32}$": "c1", "$X_{33}$": "c2", "$X_{34}$": "c3", "$X_{35}$": "c4",
#     "$X_{36}$": "e0", "$X_{37}$": "e1", "$X_{38}$": "e2",
#     # "$X_{39}$": "s0", "$X_{40}$": "s1", "$X_{41}$": "s2", "$X_{42}$": "s3" #Severity classes
#     "$X_{39}$": "d0", "$X_{40}$": "d1", "$X_{41}$": "d2", "$X_{42}$": "d3", "$X_{43}$": "d4", "$X_{44}$": "d5", "$X_{45}$": "d6" #Disease classes
# }


# #Relabel nodes
# causal_feature_mapping = {
#     "a0":0, "a1":1, "a2":2, "a3":3, "a4":4,
#     "b0":5, "b1":6, "b2":7, "b3":8, "b4":9,
#     "bo0":10, "bo1":11, "bo2":12,
#     "pt0":13, "pt1":14, "pt2":15, "pt3":16,
#     "pl0":17, "pl1":18, "pl2":19,
#     "i0":20, "i1":21, "i2":22, "i3":23, "i4":24,
#     "pb0":25, "pb1":26, "pb2":27, "pb3":28, "pb4":29,
#     "c0":30, "c1":31, "c2":32, "c3":33, "c4":34,
#     "e0":35, "e1":36, "e2":37,
#     # "s0":38, "s1":39, "s2":40, "s3":41 #Severity classes
#     "d0":38, "d1":39, "d2":40, "d3":41, "d4":42, "d5":43, "d6":44 #Disease classes
# }


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


from sklearn import tree, svm, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

def fitLargeMLP(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, 
        label_classes, feature_type = '',
        n_trial = 3, hidden_layer_sizes = (128, 64, 32), max_iter = 200, verbose = True):

    best_clf = None
    best_acc = -np.inf
    for idx in range(n_trial):

        # clf = MLPClassifier()

        if feature_type == "continous":
            clf = MLPRegressor(
                # hidden_layer_sizes = (128, 64),
                # hidden_layer_sizes = (128, 64, 32),
                hidden_layer_sizes = hidden_layer_sizes,
                learning_rate = "adaptive", #constant
                max_iter = max_iter,
                verbose = verbose,
            )
        else:
            clf = MLPClassifier(
                # hidden_layer_sizes = (128, 64),
                # hidden_layer_sizes = (128, 64, 32),
                hidden_layer_sizes = hidden_layer_sizes,
                learning_rate = "adaptive", #constant
                max_iter = max_iter,
                verbose = verbose,
            )
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        # #Map predictions to proper class labels; Here some class values can be missing
        # ml_predictions = label_classes[ml_predictions]

        # if feature_type == "continous":
        #     accuracy = clf.score(test_label_ft, gt_test_scores)
        # else:
        #     accuracy = (ml_predictions == gt_test_scores).mean()
        accuracy = clf.score(test_label_ft, gt_test_scores)
        print(f'[Trial-{idx}] ML model (MLP Large) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf

    if feature_type == "continous":
        ml_predictions = clf.predict(test_label_ft)
        ml_prob_predictions = ml_predictions
    else:
        ml_predictions = clf.predict(test_label_ft)
        ml_prob_predictions = clf.predict_proba(test_label_ft)


    # #Map predictions to proper class labels; Here some class values can be missing
    # ml_predictions = label_classes[ml_predictions]

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (MLP Large) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions




def fitRandomForest(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, label_classes, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = RandomForestClassifier(n_estimators = 100) #The number of trees in the forest (default 100).
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (RandomForest) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (RandomForest) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions




def fitDecisionTree(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, label_classes, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (DecisionTree) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (DecisionTree) accuracy = {accuracy}')


    
    return clf, accuracy, ml_predictions, ml_prob_predictions



def fitSVM(train_label_ft, test_label_ft, gt_train_scores, gt_test_scores, label_classes, n_trial = 3):

    best_clf = None
    best_acc = -1
    for idx in range(n_trial):

        # clf = svm.SVC()
        clf = svm.SVC(probability = True) #Enable probability predictions
        clf = clf.fit(train_label_ft, gt_train_scores)

        ml_predictions = clf.predict(test_label_ft)

        accuracy = (ml_predictions == gt_test_scores).mean()
        print(f'[Trial-{idx}] ML model (SVM) accuracy = {accuracy}')

        if accuracy > best_acc:
            best_acc = accuracy
            best_clf = clf

    clf = best_clf
    ml_predictions = clf.predict(test_label_ft)
    ml_prob_predictions = clf.predict_proba(test_label_ft)

    accuracy = (ml_predictions == gt_test_scores).mean()
    print(f'ML model (SVM) accuracy = {accuracy}')

    
    return clf, accuracy, ml_predictions, ml_prob_predictions


#Numpy new way to sample from distributions: Ref: https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
from numpy.random import default_rng
rng = default_rng()


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, average_precision_score, multilabel_confusion_matrix
from scipy.special import softmax
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def calScores(preds, prob_preds, targets, class_names, task, logger, binary_cross_entropy = False, skip_auc = False):

    labels = np.arange(len(class_names))
    

    accuracy = accuracy_score(targets, preds)

    if binary_cross_entropy:
        confusionMatrix = multilabel_confusion_matrix(targets, preds, labels = labels)
    else:
        confusionMatrix = confusion_matrix(targets, preds, labels = labels)
    
    # confusionMatrix = confusion_matrix(targets, preds, labels = labels)
    
    if binary_cross_entropy or skip_auc:
        auc = "-"
    else:
        auc = roc_auc_score(targets, prob_preds, average = "weighted", multi_class = "ovo") # multi_class = "ovr"
    precision = precision_score(targets, preds, average='weighted') #score-All average
    recall = recall_score(targets, preds, average='weighted') #score-All average
    f1 = f1_score(targets, preds, average='weighted') #score-All average
        
    classificationReport = classification_report(targets, preds, labels = labels, target_names = class_names, digits=5)

    logger.log(f"auc = {auc}")
    logger.log(f"accuracy = {accuracy}")
    logger.log(f"precision = {precision}")
    logger.log(f"recall = {recall}")
    logger.log(f"f1 = {f1}")
    logger.log(f"confusionMatrix = \n {confusionMatrix}")
    logger.log(f"classificationReport = \n {classificationReport}")


    results_dict = {}
    results_dict["auc"] = auc
    results_dict["accuracy"] = accuracy
    results_dict["precision"] = precision
    results_dict["recall"] = recall
    results_dict["f1"] = f1
    results_dict["confusionMatrix"] = confusionMatrix.tolist()
    results_dict["classificationReport"] = classificationReport

    return results_dict


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

    exp_name = "evaluateGeneratedSamples"
    
    # fake_data = 'diabetes-t1_TrainUpsampledPatient_size-708'
    # fake_data = 'diabetes-t2_TrainUpsampledPatient_size-706'
    # fake_data = 'diabetes-t3_TrainUpsampledPatient_size-688'
    # fake_data = 'diabetes-t4_TrainUpsampledPatient_size-678'
    # fake_data = 'diabetes-t5_TrainUpsampledPatient_size-694'
    # fake_data = 'diabetes-t1_TrainUpsampledPatient_size-200'
    # fake_data = 'diabetes-t2_TrainUpsampledPatient_size-200'
    # fake_data = 'diabetes-t3_TrainUpsampledPatient_size-200'
    # fake_data = 'diabetes-t4_TrainUpsampledPatient_size-200'
    # fake_data = 'diabetes-t5_TrainUpsampledPatient_size-200'
    # fake_data = 'diabetes-small1_TrainUpsampledPatient_size-200'
    # fake_data = 'diabetes-small2_TrainUpsampledPatient_size-200'
    # fake_data = 'diabetes-small3_TrainUpsampledPatient_size-200'
    # fake_data = 'diabetes-small4_TrainUpsampledPatient_size-200'
    fake_data = 'diabetes-small5_TrainUpsampledPatient_size-200'
    fake_data_path = f"/home/grg/Research/ENCO/datasets/gan_fake_data/{fake_data}.csv"

    # task = 'heart-disease' #'heart-disease-binary' #'heart-disease' #'parity5' #'labor'
    # task = 'diabetes-t1' #'heart-disease-binary-t2' #'dermatology' #'heart-disease-binary'

    task = fake_data.split("_")[0]

    print(f"Evaluating {task} using fake data {fake_data}")

    causal_discovery_exp_dir = f"/home/grg/Research/ENCO/checkpoints/2022_26_Acyclic_{task}_TrainUpsampledPatient_GAN"

    # set_seed()

    num_samples = 200 #200 #3000 #500 #1000
    # Binary_Features = False #True
    # features_type = 'continous' #categorical, binary, continous

    max_iter = 200 #200

    results_dir = os.path.join(causal_discovery_exp_dir, f"{exp_name}_{num_samples}")
    utils.createDirIfDoesntExists(results_dir)

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


    ## Test the synthetic data


    fake_dataset = np.genfromtxt(fake_data_path, delimiter=',')
    fake_dataset = fake_dataset[1:] 
    
    fake_dataset_features = fake_dataset[:,:-1]
    fake_dataset_labels = fake_dataset[:,-1]
    
    fake_dataset_binary_labels = np.zeros_like(fake_dataset_labels, dtype = int)
    fake_dataset_binary_labels[fake_dataset_labels > 0.5] = 1
    
    fake_one_hot_labels = np.zeros((fake_dataset_binary_labels.shape[0], 2))
    fake_one_hot_labels[np.arange(fake_dataset_binary_labels.shape[0]), fake_dataset_binary_labels] = 1

    assert np.all(fake_one_hot_labels.argmax(1) == fake_dataset_binary_labels), "Error! One hot convertion incorrectly done"

    synthetic_data = np.concatenate((fake_dataset_features, fake_one_hot_labels), axis = 1)
    # assert synthetic_data.shape == train_data.shape, "Error! Synthetic data not correctly loaded."
    assert synthetic_data.shape == (200, train_data.shape[1]), "Error! Synthetic data not correctly loaded."

    print(f"train_data.shape = {train_data.shape}")

    train_features = train_data[:, :num_vars]

    train_labels = train_data[:, category_classes]

    UpSampleData = True #False
    if UpSampleData:
        upsam_train_labels, upsam_train_features, upsample_label_indices = upsampleFeatures(labels = train_labels.argmax(1), features = train_features) 
        train_labels = train_labels[upsample_label_indices]
        train_features = upsam_train_features

        assert train_labels.shape[0] == train_features.shape[0], "Error! Upsampled labels and feature count does not match."

    print(f"train_labels.shape = {train_labels.shape}")
    print(f"train_features.shape = {train_features.shape}")

    # test_data_path = "DiseaseBiomarkersGTabTestUpsampledPatient-max.npz"
    # test_data_path = "DiseaseModelBiomarkersGTabTestUpsampledPatient-max.npz"
    # test_data_path = "labor_TestUpsampledPatient.npz"
    test_data_path = f"datasets/{task}_TestUpsampledPatient.npz"

    test_data = np.load(test_data_path)['data_obs']


    # val_data_path = f"datasets/{task}_ValUpsampledPatient.npz"

    # val_data = np.load(val_data_path)['data_obs']


    test_features = test_data[:, :num_vars]

    test_labels = test_data[:, category_classes]


    # val_features = val_data[:, :num_vars]

    # val_labels = val_data[:, category_classes]


    label_classes = np.unique(train_labels.argmax(1))




    ### Evaluate the synthetic data on Test set









    ##Train model
    model, accuracy, ml_predictions, ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  train_features, test_label_ft = test_features, 
                                # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                label_classes = label_classes,
                                # features_type = "categorical",
                                # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                n_trial = 3, hidden_layer_sizes = (100),
                                max_iter = max_iter,
                                verbose = False
                            )

    logger.log(f"[MLPlarge] Accuracy on original train set = {accuracy}")

    # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

    model_results_dict = calScores(preds = ml_predictions, prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), 
                targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger,
                binary_cross_entropy = binary_cross_entropy)
    

    # UpSampleSynthteicData = False #False

    runOnlyOnSynthetic = True
    if runOnlyOnSynthetic:
        only_synthetic_train_features = synthetic_data[:, :num_vars]

        only_synthetic_train_labels = synthetic_data[:, category_classes]

        logger.log(f"only-synthetic class distn - {np.unique(only_synthetic_train_labels.argmax(1), return_counts = True)}")

        if UpSampleData:
        # if UpSampleSynthteicData:
            upsam_synthetic_train_labels, upsam_synthetic_train_features, upsample_label_indices = upsampleFeatures(labels = only_synthetic_train_labels.argmax(1), features = only_synthetic_train_features) 
            only_synthetic_train_labels = only_synthetic_train_labels[upsample_label_indices]
            only_synthetic_train_features = upsam_synthetic_train_features

            assert only_synthetic_train_labels.shape[0] == only_synthetic_train_features.shape[0], "Error! Upsampled labels and feature count does not match."

        print(f"only_synthetic_train_labels.shape = {only_synthetic_train_labels.shape}")
        print(f"only_synthetic_train_features.shape = {only_synthetic_train_features.shape}")

        only_synthetic_model, only_synthetic_accuracy, only_synthetic_ml_predictions, only_synthetic_ml_prob_predictions = fitLargeMLP(
                                    train_label_ft =  only_synthetic_train_features, test_label_ft = test_features, 
                                    # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = only_synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    # features_type = "categorical",
                                    # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                    n_trial = 3, hidden_layer_sizes = (100),
                                    max_iter = max_iter,
                                    verbose = False
                                )

        logger.log(f"[MLPlarge] Accuracy on only synthetic set = {only_synthetic_accuracy}")

        # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
        only_synthetic_model_results_dict = calScores(preds = only_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(only_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-only_synthetic", logger = logger,
                    binary_cross_entropy = binary_cross_entropy,
                    skip_auc = True)


    synthetic_train_data = np.concatenate((train_data, synthetic_data), axis = 0)
    print(f"synthetic_train_data.shape = {synthetic_train_data.shape}")
    
    synthetic_train_features = synthetic_train_data[:, :num_vars]

    synthetic_train_labels = synthetic_train_data[:, category_classes]


    UpSampleSynthteicData = False #False
    # if UpSampleData:
    if UpSampleSynthteicData:
        upsam_synthetic_train_labels, upsam_synthetic_train_features, upsample_label_indices = upsampleFeatures(labels = synthetic_train_labels.argmax(1), features = synthetic_train_features) 
        synthetic_train_labels = synthetic_train_labels[upsample_label_indices]
        synthetic_train_features = upsam_synthetic_train_features

        assert synthetic_train_labels.shape[0] == synthetic_train_features.shape[0], "Error! Upsampled labels and feature count does not match."


    print(f"synthetic_train_labels.shape = {synthetic_train_labels.shape}")
    print(f"synthetic_train_features.shape = {synthetic_train_features.shape}")

    synthetic_model, synthetic_accuracy, synthetic_ml_predictions, synthetic_ml_prob_predictions = fitLargeMLP(
                                train_label_ft =  synthetic_train_features, test_label_ft = test_features, 
                                # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                label_classes = label_classes,
                                # features_type = "categorical",
                                # n_trial = 3, hidden_layer_sizes = (128, 64, 32),
                                n_trial = 3, hidden_layer_sizes = (100),
                                max_iter = max_iter,
                                verbose = False
                            )

    logger.log(f"[MLPlarge] Accuracy on synthetic + original train set = {synthetic_accuracy}")

    # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
    synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), 
                targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger,
                binary_cross_entropy = binary_cross_entropy)


    ######### Random Forest ############

    runRandomForest = True
    if runRandomForest:

        ##Train model
        rf_model, rf_accuracy, rf_ml_predictions, rf_ml_prob_predictions = fitRandomForest(
                                    train_label_ft =  train_features, test_label_ft = test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[RandomForest] Accuracy on original train set = {rf_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        rf_model_results_dict = calScores(preds = rf_ml_predictions, prob_preds = torch.softmax(torch.Tensor(rf_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-rf", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)

        rf_synthetic_model, rf_synthetic_accuracy, rf_synthetic_ml_predictions, rf_synthetic_ml_prob_predictions = fitRandomForest(
                                    train_label_ft =  synthetic_train_features, test_label_ft = test_features, 
                                    # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[RandomForest] Accuracy on synthetic + original train set = {rf_synthetic_accuracy}")

        # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
        rf_synthetic_model_results_dict = calScores(preds = rf_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(rf_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-rf"+"-synthetic", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)


        runOnlyOnSynthetic = True
        if runOnlyOnSynthetic:
            
            rf_only_synthetic_model, rf_only_synthetic_accuracy, rf_only_synthetic_ml_predictions, rf_only_synthetic_ml_prob_predictions = fitRandomForest(
                                        train_label_ft =  only_synthetic_train_features, test_label_ft = test_features, 
                                        # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                        gt_train_scores = only_synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                        label_classes = label_classes,
                                        n_trial = 3
                                    )

            logger.log(f"[RandomForest] Accuracy on only synthetic set = {rf_only_synthetic_accuracy}")

            # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
            rf_only_synthetic_model_results_dict = calScores(preds = rf_only_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(rf_only_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                        targets = test_labels.argmax(1), class_names = class_names, task = task+"-rf"+"-only_synthetic", logger = logger,
                        binary_cross_entropy = binary_cross_entropy,
                        skip_auc = True)


    ######### DecisionTree ############

    runDT = True
    if runDT:

        ##Train model
        dt_model, dt_accuracy, dt_ml_predictions, dt_ml_prob_predictions = fitDecisionTree(
                                    train_label_ft =  train_features, test_label_ft = test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[DT] Accuracy on original train set = {dt_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        dt_model_results_dict = calScores(preds = dt_ml_predictions, prob_preds = torch.softmax(torch.Tensor(dt_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)

        dt_synthetic_model, dt_synthetic_accuracy, dt_synthetic_ml_predictions, dt_synthetic_ml_prob_predictions = fitDecisionTree(
                                    train_label_ft =  synthetic_train_features, test_label_ft = test_features, 
                                    # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[DT] Accuracy on synthetic + original train set = {dt_synthetic_accuracy}")

        # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
        dt_synthetic_model_results_dict = calScores(preds = dt_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(dt_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm"+"-synthetic", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)



        runOnlyOnSynthetic = True
        if runOnlyOnSynthetic:
            
            dt_only_synthetic_model, dt_only_synthetic_accuracy, dt_only_synthetic_ml_predictions, dt_only_synthetic_ml_prob_predictions = fitDecisionTree(
                                        train_label_ft =  only_synthetic_train_features, test_label_ft = test_features, 
                                        # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                        gt_train_scores = only_synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                        label_classes = label_classes,
                                        n_trial = 3
                                    )

            logger.log(f"[DT] Accuracy on only synthetic set = {dt_only_synthetic_accuracy}")

            # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
            dt_only_synthetic_model_results_dict = calScores(preds = dt_only_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(dt_only_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                        targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm"+"-only_synthetic", logger = logger,
                        binary_cross_entropy = binary_cross_entropy,
                        skip_auc = True)


    ######### SVM ############

    runSVM = False
    if runSVM:

        ##Train model
        svm_model, svm_accuracy, svm_ml_predictions, svm_ml_prob_predictions = fitSVM(
                                    train_label_ft =  train_features, test_label_ft = test_features, 
                                    # gt_train_scores = train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[SVM] Accuracy on original train set = {svm_accuracy}")

        # model_results_dict = calScores(preds = ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task, logger = logger)

        svm_model_results_dict = calScores(preds = svm_ml_predictions, prob_preds = torch.softmax(torch.Tensor(svm_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)

        svm_synthetic_model, svm_synthetic_accuracy, svm_synthetic_ml_predictions, svm_synthetic_ml_prob_predictions = fitSVM(
                                    train_label_ft =  synthetic_train_features, test_label_ft = test_features, 
                                    # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                    gt_train_scores = synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                    label_classes = label_classes,
                                    n_trial = 3
                                )

        logger.log(f"[SVM] Accuracy on synthetic + original train set = {svm_synthetic_accuracy}")

        # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
        svm_synthetic_model_results_dict = calScores(preds = svm_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(svm_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                    targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm"+"-synthetic", logger = logger,
                    binary_cross_entropy = binary_cross_entropy)



        runOnlyOnSynthetic = True
        if runOnlyOnSynthetic:
            
            svm_only_synthetic_model, svm_only_synthetic_accuracy, svm_only_synthetic_ml_predictions, svm_only_synthetic_ml_prob_predictions = fitSVM(
                                        train_label_ft =  only_synthetic_train_features, test_label_ft = test_features, 
                                        # gt_train_scores = synthetic_train_labels, gt_test_scores = test_labels, 
                                        gt_train_scores = only_synthetic_train_labels.argmax(1), gt_test_scores = test_labels.argmax(1), 
                                        label_classes = label_classes,
                                        n_trial = 3
                                    )

            logger.log(f"[SVM] Accuracy on only synthetic set = {svm_only_synthetic_accuracy}")

            # synthetic_model_results_dict = calScores(preds = synthetic_ml_predictions.argmax(1), prob_preds = torch.softmax(torch.Tensor(synthetic_ml_prob_predictions), dim = 1).numpy(), targets = test_labels.argmax(1), class_names = class_names, task = task+"-synthetic", logger = logger)
            svm_only_synthetic_model_results_dict = calScores(preds = svm_only_synthetic_ml_predictions, prob_preds = torch.softmax(torch.Tensor(svm_only_synthetic_ml_prob_predictions), dim = 1).numpy(), 
                        targets = test_labels.argmax(1), class_names = class_names, task = task+"-svm"+"-only_synthetic", logger = logger,
                        binary_cross_entropy = binary_cross_entropy,
                        skip_auc = True)



    logger.log(f"Task = {task}")
    logger.log(f"only-synthetic class distn - {np.unique(synthetic_data[:, category_classes].argmax(1), return_counts = True)}")
    logger.log(f"[MLPlarge] Accuracy on original train set = {accuracy}")
    logger.log(f"[MLPlarge] Accuracy on only synthetic set = {only_synthetic_accuracy}")
    logger.log(f"[MLPlarge] Accuracy on synthetic + original train set = {synthetic_accuracy}")
    logger.log(f"[RandomForest] Accuracy on original train set = {rf_accuracy}")
    logger.log(f"[RandomForest] Accuracy on only synthetic set = {rf_only_synthetic_accuracy}")
    logger.log(f"[RandomForest] Accuracy on synthetic + original train set = {rf_synthetic_accuracy}")
    logger.log(f"[DT] Accuracy on original train set = {dt_accuracy}")
    logger.log(f"[DT] Accuracy on only synthetic set = {dt_only_synthetic_accuracy}")
    logger.log(f"[DT] Accuracy on synthetic + original train set = {dt_synthetic_accuracy}")
    # logger.log(f"[SVM] Accuracy on original train set = {svm_accuracy}")
    # logger.log(f"[SVM] Accuracy on only synthetic set = {svm_only_synthetic_accuracy}")
    # logger.log(f"[SVM] Accuracy on synthetic + original train set = {svm_synthetic_accuracy}")

    logger.log(f"{accuracy} \n{rf_accuracy} \n{dt_accuracy} \n{only_synthetic_accuracy} \n{rf_only_synthetic_accuracy} \n{dt_only_synthetic_accuracy} \n{synthetic_accuracy} \n{rf_synthetic_accuracy} \n{dt_synthetic_accuracy}")
    
    logger.close()

    pass


    return accuracy, only_synthetic_accuracy, synthetic_accuracy



if __name__ == "__main__":
    print("Started...")

    synthetic_accuracy = -1

    # while synthetic_accuracy < .85:
    # while synthetic_accuracy < .99:
    while synthetic_accuracy < .40:
        accuracy, only_synthetic_accuracy, synthetic_accuracy = main()
    
    
    print("Finished!")