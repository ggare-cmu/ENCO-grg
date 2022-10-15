from dataclasses import dataclass
from termios import VDISCARD
from matplotlib import test
import numpy as np

import utils


import pandas as pd 


# pip install pmlb - Penn Machine Learning Benchmark data repository
import pmlb
from pmlb import fetch_data, classification_dataset_names, regression_dataset_names


# pip install uci-dataset - https://github.com/maryami66/uci_dataset/blob/main/lists/classification_dataset.md
import uci_dataset

from sklearn.model_selection import train_test_split




def getClassificationData(classification_dataset, plml_dataset = True):

    if classification_dataset == "cifar":
        train_feat = np.load("/home/grg/Research/ENCO/datasets_cifar/cifar10_f15_feat_train.npy")
        test_feat = np.load("/home/grg/Research/ENCO/datasets_cifar/cifar10_f15_feat_test.npy")
        train_target = np.load("/home/grg/Research/ENCO/datasets_cifar/cifar10_f15_target_train.npy")
        test_target = np.load("/home/grg/Research/ENCO/datasets_cifar/cifar10_f15_target_test.npy")

    class_name_dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    class_id_to_name_dict = {}
    for n,i in class_name_dict.items():
        class_id_to_name_dict[i] = n

    class_names = np.unique(test_target)
    assert np.array_equal(class_names, np.arange(class_names.shape[0])), f"Error! Class names are incorrect : class_names = {class_names}"
    class_names = [class_id_to_name_dict[i] for i in class_names]

    num_classes = len(class_names)

    train_target_one_hot = np.zeros((train_target.shape[0], num_classes))
    train_target_one_hot[np.arange(train_target.shape[0]), train_target] = 1

    assert np.all(train_target_one_hot.argmax(1) == train_target), "Error! One hot convertion incorrectly done"

    test_target_one_hot = np.zeros((test_target.shape[0], num_classes))
    test_target_one_hot[np.arange(test_target.shape[0]), test_target] = 1

    assert np.all(test_target_one_hot.argmax(1) == test_target), "Error! One hot convertion incorrectly done"
    
    num_input_features = train_feat.shape[1]
    vars_list = [f'x{i+1}' for i in range(num_input_features)]

    # class_list = [f'c{i+1}' for i in range(num_classes)]
    class_list = class_names
    features_list = vars_list + class_list


    ## Set variable feature type 

    feature_type = {}
    for var, x in zip(vars_list, train_feat.T):
        
        var_unique = np.unique(x).tolist()
        
        if var_unique == [0,1]:
            var_type = 'binary'
        elif len(var_unique) <= 20:
            var_type = 'categorical'
        elif min(var_unique) == 0 and max(var_unique) == 1:
            var_type = 'continous'
        else:
            var_type = 'continous'
        
        # print(f"{var} feature type set as {var_type}; np.unique({var}) = {var_unique}")
        print(f"{var} feature type set as {var_type}; np.unique({var}) = {var_unique[:5]}")
        
        feature_type[var] = var_type
        

    for cls in class_list:
        # feature_type[cls] = 'binary'
        feature_type[cls] = 'binary-class'

    print(f"All vars feature type = {feature_type}")

    # train_ft, test_ft, train_label, test_label = train_test_split(X, y, test_size = 0.3)
    # train_ft, test_ft, train_label, test_label = train_test_split(X, y_one_hot, test_size = 0.3)
    # train_ft, test_ft, train_label, test_label = train_test_split(X, y_one_hot, test_size = 0.2)
    # train_ft, val_ft, train_label, val_label = train_test_split(train_ft, train_label, test_size = 0.125)


    train_ft, test_ft, train_label, test_label = train_feat, test_feat, train_target_one_hot, test_target_one_hot
    train_ft, val_ft, train_label, val_label = train_test_split(train_ft, train_label, test_size = 0.125)

    # assert np.array_equal(np.unique(X), np.unique(train_ft)), "Error! Non-uniform data-splitting."
    # assert np.array_equal(np.unique(X), np.unique(test_ft)), "Error! Non-uniform data-splitting."
    assert np.array_equal(np.unique(test_target), np.unique(train_label.argmax(1))), "Error! Non-uniform data-splitting."
    assert np.array_equal(np.unique(test_target), np.unique(test_label.argmax(1))), "Error! Non-uniform data-splitting."
    assert np.array_equal(np.unique(test_target), np.unique(val_label.argmax(1))), "Error! Non-uniform data-splitting."

    return train_ft, test_ft, val_ft, train_label, test_label, val_label, class_names, num_classes, vars_list, class_list, num_input_features, feature_type




def upsampleFeatures(labels_one_hot, features):

    labels = labels_one_hot.argmax(1)

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

    # upsampled_labels = labels[label_indices]
    upsampled_labels_one_hot = labels_one_hot[label_indices]

    upsampled_labels = upsampled_labels_one_hot.argmax(1)

    classes, count = np.unique(upsampled_labels, return_counts = True)
    print(f"[Post-Upsampling] classes, count = {classes, count}")   

    assert np.array_equal(count, max_count * np.ones(len(classes))), "Error! Upsampling didn't result in class-balance"

    assert upsampled_labels_one_hot.shape[0] == upsampled_features.shape[0], "Error! Labels incorrectly upsampled."

    # return upsampled_labels, upsampled_features, upsample_label_indices
    return upsampled_labels_one_hot, upsampled_features, upsample_label_indices





def main():

    ##PLML-Datasets
    # classification_dataset = "parity5" #"labor" #"diabetes" #"diabetes" #"breast_cancer_wisconsin"
    
    ##UCI-Datasets
    classification_dataset = "cifar"
    trial = "t1"

    plml_dataset = False

    train_ft, test_ft, val_ft, train_label, test_label, val_label, class_names, num_classes, vars_list, class_list, num_input_features, feature_type = getClassificationData(classification_dataset, plml_dataset)


    upsampleTrainFeatures = True
    
    #Upsample train set for class balancing
    if upsampleTrainFeatures:

        upsam_train_labels, upsam_train_ft, upsample_label_indices = upsampleFeatures(labels_one_hot = train_label, features = train_ft) 

        train_ft = upsam_train_ft
        train_label = upsam_train_labels


    train_biomarkers = np.concatenate((train_ft, train_label), axis = 1)
    assert train_biomarkers.shape == (train_ft.shape[0], train_ft.shape[1] + train_label.shape[1]), "Error! Features concatenation incorrect"


    test_biomarkers = np.concatenate((test_ft, test_label), axis = 1)
    assert test_biomarkers.shape == (test_ft.shape[0], test_ft.shape[1] + test_label.shape[1]), "Error! Features concatenation incorrect"


    val_biomarkers = np.concatenate((val_ft, val_label), axis = 1)
    assert val_biomarkers.shape == (val_ft.shape[0], val_ft.shape[1] + val_label.shape[1]), "Error! Features concatenation incorrect"



    # train_biomarkers = np.array(train_biomarkers, dtype = data_type)
    # np.save(f"datasets/{classification_dataset}_data2trainUpsampled.npy", train_biomarkers)

    # test_biomarkers = np.array(test_biomarkers, dtype = data_type)
    # np.save(f"datasets/{classification_dataset}_data3testUpsampled.npy", test_biomarkers)

    ##Set data type

    # data_type = np.int32
    # data_type = np.uint8

    all_feature_types = list(feature_type.values())
    if 'continous' in all_feature_types:
        data_type = np.float32
    elif np.array_equal(np.unique(all_feature_types), ['binary', 'binary-class']):
        data_type = np.uint8
    else:
        data_type = np.int32

    print(f"Using data type = {data_type}")

    train_biomarkers = train_biomarkers.astype(data_type)
    test_biomarkers = test_biomarkers.astype(data_type)
    val_biomarkers = val_biomarkers.astype(data_type)
    
    adj_matrix = np.zeros((test_biomarkers.shape[1], test_biomarkers.shape[1]), dtype=np.uint8)

    data_int = np.zeros((test_biomarkers.shape[1], 1, test_biomarkers.shape[1]), dtype=data_type)

    train_biomarkers_csv = np.concatenate((train_biomarkers[:,:-num_classes],train_biomarkers[:,-num_classes:].argmax(1)[:, np.newaxis]), axis = 1)
    np.savetxt(f"datasets/{classification_dataset}-{trial}_TrainUpsampledPatient.csv", train_biomarkers_csv, delimiter = ",")


    np.savez(f"datasets/{classification_dataset}-{trial}_TrainUpsampledPatient.npz", data_obs = train_biomarkers, data_int = data_int, adj_matrix = adj_matrix, 
            vars_list = vars_list, class_list = class_list, feature_type = feature_type)

    np.savez(f"datasets/{classification_dataset}-{trial}_TestUpsampledPatient.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix, 
            vars_list = vars_list, class_list = class_list, feature_type = feature_type)

    np.savez(f"datasets/{classification_dataset}-{trial}_ValUpsampledPatient.npz", data_obs = val_biomarkers, data_int = data_int, adj_matrix = adj_matrix, 
            vars_list = vars_list, class_list = class_list, feature_type = feature_type)




'''
If your causal graph/dataset is specified in a .bif format as the real-world graphs, you can directly start an experiment on it using experiments/run_exported_graphs.py. The alternative format is a .npz file which contains a observational and interventional dataset. The file needs to contain the following keys:

data_obs: A dataset of observational samples. This array must be of shape [M, num_vars] where M is the number of data points. For categorical data, it should be any integer data type (e.g. np.int32 or np.uint8).
data_int: A dataset of interventional samples. This array must be of shape [num_vars, K, num_vars] where K is the number of data points per intervention. The first axis indicates the variables on which has been intervened to gain this dataset.
adj_matrix: The ground truth adjacency matrix of the graph (shape [num_vars, num_vars], type bool or integer). The matrix is used to determine metrics like SHD during/after training. If the ground truth matrix is not known, you can submit a zero-matrix (keep in mind that the metrics cannot be used in this case).
'''
if __name__ == "__main__":
    print(f"Started...")
    main()
    print(f"Finished!")
