import numpy as np

import utils



label_names = [ 
                'al-none', 'al-weak', 'al-bold', 'al-*also* stacked', 'al-*also* wide (> 2cm)',
                'bl-none', 'bl-few (1-3)', 'bl-some (4-6)', 'bl-many|coalescing', "bl-\"white\" (no striations)",   
                'bo-N/A', 'bo-pleura', 'bo-sub-plu', 
                'pt-<1mm', 'pt-2-3mm', 'pt-4-5mm', 'pt->5mm',
                'pl-top', 'pl-mid', 'pl-btm', 
                'pi-none', 'pi-<5mm (few)', 'pi-<5mm (multiple)', 'pi-5-10mm', 'pi->10mm', 
                'pb-none', 'pb-<5mm (few)', 'pb-<5mm (multiple)', 'pb-5-10mm', 'pb->10mm',
                'cn-none', 'cn-<5mm (few)', 'cn-<5mm (multiple)', 'cn-5-10mm', 'cn->10mm',
                'ef-none', 'ef-<5mm', 'ef->5mm', 
            ]

def main():

    labels_path = "/home/grg/Research/ENCO/user_label_ggare_2.json"

    label_dict = utils.readJson(labels_path)

    model_train_labels_path = "/home/grg/Research/ENCO/user_label_tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_C_train.json"

    model_test_labels_path = "/home/grg/Research/ENCO/user_label_tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_C_test_updated.json"

    model_train_labels = utils.readJson(model_train_labels_path)
    model_test_labels = utils.readJson(model_test_labels_path)

    ## Read Data folds 
    data_split_path = "/home/grg/Research/ENCO/dataset_split_equi_class_R1.json"
    data_split_dict = utils.readJson(data_split_path)

    train_folds = ['A', 'B']
    test_folds = ['D']

    train_videos = []
    [train_videos.extend(data_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in train_folds]
    
    test_videos = []
    [test_videos.extend(data_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in test_folds]


    features = ['alines', 'blines', 'blines_origin', 'pleural_thickness', 'pleural_location', 'pleural_indent', 'pleural_break', 'consolidation', 'effusion', ]

    # features = features + ['lung-severity']

    # biomarkers = []
    # for video, label in label_dict.items():
    #     v_bio = np.hstack([label[f] for f in features])
    #     biomarkers.append(v_bio)

    USE_PROB = False
    
    train_biomarkers = []
    for video in train_videos:
        
        model_label = model_train_labels[video]
        v_bio =[model_label[f] for f in features]
        assert np.hstack(v_bio).shape[0] == 38, "Error! Model labels incorrectly loaded."

        #Covert Prob to Class - using threshold 0.5
        if not USE_PROB:
            v_bio = np.hstack(v_bio)
            v_bio = 1*(v_bio > 0.5)
            v_bio = v_bio.tolist()

        label = label_dict[video]
        v_bio.append(label['lung-severity'])

        v_bio = np.hstack(v_bio)
        assert v_bio.shape[0] == 42, "Error! GroundTruth labels incorrectly loaded."
        
        train_biomarkers.append(v_bio)

    
    test_biomarkers = []
    for video in test_videos:
        
        model_label = model_test_labels[video]
        v_bio =[model_label[f] for f in features]
        assert np.hstack(v_bio).shape[0] == 38, "Error! Model labels incorrectly loaded."

        #Covert Prob to Class - using threshold 0.5
        if not USE_PROB:
            v_bio = np.hstack(v_bio)
            v_bio = 1*(v_bio > 0.5)
            v_bio = v_bio.tolist()

        label = label_dict[video]
        v_bio.append(label['lung-severity'])

        v_bio = np.hstack(v_bio)
        assert v_bio.shape[0] == 42, "Error! GroundTruth labels incorrectly loaded."
        
        test_biomarkers.append(v_bio)



    #Save as npz
    train_biomarkers = np.array(train_biomarkers, dtype=np.uint8)
    np.save("data2trainbio.npy", train_biomarkers)

    test_biomarkers = np.array(test_biomarkers, dtype=np.uint8)
    np.save("data3testbio.npy", test_biomarkers)


    no_features = train_biomarkers.shape[1]

    adj_matrix = np.zeros((no_features, no_features), dtype=np.uint8)

    data_int = np.zeros((no_features, 1, no_features), dtype=np.uint8)

    np.savez("ModelBiomarkersTrain.npz", data_obs = train_biomarkers, data_int = data_int, adj_matrix = adj_matrix)

    np.savez("ModelBiomarkersTest.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix)

    


'''
f your causal graph/dataset is specified in a .bif format as the real-world graphs, you can directly start an experiment on it using experiments/run_exported_graphs.py. The alternative format is a .npz file which contains a observational and interventional dataset. The file needs to contain the following keys:

data_obs: A dataset of observational samples. This array must be of shape [M, num_vars] where M is the number of data points. For categorical data, it should be any integer data type (e.g. np.int32 or np.uint8).
data_int: A dataset of interventional samples. This array must be of shape [num_vars, K, num_vars] where K is the number of data points per intervention. The first axis indicates the variables on which has been intervened to gain this dataset.
adj_matrix: The ground truth adjacency matrix of the graph (shape [num_vars, num_vars], type bool or integer). The matrix is used to determine metrics like SHD during/after training. If the ground truth matrix is not known, you can submit a zero-matrix (keep in mind that the metrics cannot be used in this case).
'''
if __name__ == "__main__":
    print(f"Started...")
    main()
    print(f"Finished!")
