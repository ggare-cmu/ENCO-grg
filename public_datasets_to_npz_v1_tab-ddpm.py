from dataclasses import dataclass
from termios import VDISCARD
import numpy as np

import utils


import pandas as pd 


# pip install pmlb - Penn Machine Learning Benchmark data repository
import pmlb
from pmlb import fetch_data, classification_dataset_names, regression_dataset_names


# pip install uci-dataset - https://github.com/maryami66/uci_dataset/blob/main/lists/classification_dataset.md
import uci_dataset

from sklearn.model_selection import train_test_split




def getClassificationData(dataset_path):

   

    train_X = np.load(os.path.join(dataset_path, "X_num_train.npy")) 
    train_y = np.load(os.path.join(dataset_path, "y_train.npy")) 
    val_X = np.load(os.path.join(dataset_path, "X_num_val.npy")) 
    val_y = np.load(os.path.join(dataset_path, "y_val.npy")) 
    test_X = np.load(os.path.join(dataset_path, "X_num_test.npy")) 
    test_y = np.load(os.path.join(dataset_path, "y_test.npy")) 

            

    class_names = np.unique(test_y)
    assert np.array_equal(class_names, np.arange(class_names.shape[0])), f"Error! Class names are incorrect : class_names = {class_names}"

    num_classes = len(class_names)

    def convertToOneHot(y):
        y_one_hot = np.zeros((y.shape[0], num_classes))
        y_one_hot[np.arange(y.shape[0]), y] = 1

        assert np.all(y_one_hot.argmax(1) == y), "Error! One hot convertion incorrectly done"

        return y_one_hot

    train_y_one_hot = convertToOneHot(train_y)
    val_y_one_hot = convertToOneHot(val_y)
    test_y_one_hot = convertToOneHot(test_y)

    num_input_features = train_X.shape[1]
    vars_list = [f'x{i+1}' for i in range(num_input_features)]

    class_list = [f'c{i+1}' for i in range(num_classes)]
    features_list = vars_list + class_list


    ## Set variable feature type 

    #Normalize the features to [0,1] range
    def normalizeFeature(X):
        print(f"[Pre-normalization] X features range = {X.mean(0)}")
        X_min = X.min(0)
        X_max = X.max(0)
        X = (X - X_min)/(X_max - X_min)
        print(f"[Post-normalization] X features range = {X.mean(0)}")
        return X


    train_X = normalizeFeature(train_X)
    val_X = normalizeFeature(val_X)
    test_X = normalizeFeature(test_X)


    train_X_min = train_X.min(0)
    train_X_max = train_X.max(0)

    feature_type = {}
    for var, x, x_min, x_max in zip(vars_list, train_X.T, train_X_min, train_X_max):
        
        var_unique = np.unique(x).tolist()
        
        # if var_unique == [0,1]:
        #     var_type = 'binary'
        # elif len(var_unique) <= 20:
        #     var_type = 'categorical'
        # elif min(var_unique) == 0 and max(var_unique) == 1:
        #     var_type = 'continous'
        # else:
        #     var_type = 'continous'


        var_type = 'continous'
        
        # print(f"{var} feature type set as {var_type}; np.unique({var}) = {var_unique}")
        
        feature_type[var] = var_type
        feature_type[var+"_min"] = x_min
        feature_type[var+"_max"] = x_max
        

    for cls in class_list:
        # feature_type[cls] = 'binary'
        feature_type[cls] = 'binary-class'

    print(f"All vars feature type = {feature_type}")

    train_ft, test_ft, val_ft = train_X, test_X, val_X
    train_label, test_label, val_label = train_y_one_hot, test_y_one_hot, val_y_one_hot

    # assert np.array_equal(np.unique(X), np.unique(train_ft)), "Error! Non-uniform data-splitting."
    # assert np.array_equal(np.unique(X), np.unique(test_ft)), "Error! Non-uniform data-splitting."
    assert np.array_equal(np.unique(train_y), np.unique(train_label.argmax(1))), "Error! Non-uniform data-splitting."
    assert np.array_equal(np.unique(val_y), np.unique(val_label.argmax(1))), "Error! Non-uniform data-splitting."
    assert np.array_equal(np.unique(test_y), np.unique(test_label.argmax(1))), "Error! Non-uniform data-splitting."

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




import os

def main():

    ##Tab-DDPM-Datasets
    dataset_root_path = "/home/grg/Research/ENCO-grg/ddataset_tab-ddpm/"

    classification_dataset = "tab-ddpm-higgs-small" 
    
    trial = "t1"

    dataset_path = os.path.join(dataset_root_path, classification_dataset)
    
    train_ft, test_ft, val_ft, train_label, test_label, val_label, class_names, num_classes, vars_list, class_list, num_input_features, feature_type = getClassificationData(dataset_path)


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
