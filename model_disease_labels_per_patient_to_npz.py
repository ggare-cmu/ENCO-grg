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


import pandas as pd 

def getPatientData(excel_file_path):

    patient_data = pd.read_excel(excel_file_path)
    header = list(patient_data.columns)
    patient_data = np.array(patient_data)
    
    #Get videos ids where SF ratio is empty
    non_sf_videos = np.argwhere(np.isnan(patient_data[:, header.index('S/F on date of exam')].astype(np.float32)))
    assert np.isnan(patient_data[non_sf_videos, header.index('S/F on date of exam')].astype(np.float32)).all(), "Error! Wrong indices picked."

    #Exluded non SF data videos
    patient_data = np.delete(patient_data, non_sf_videos, axis = 0)

    assert not np.isnan(patient_data[:, header.index('S/F on date of exam')].astype(np.float32)).any(), "Error! Wrong indices removed."

    return header, patient_data


'''
Method that aggreates patient-scan_date wise video features and consolidates into a single features.
consolidation_types:
    1) max = Take max of every feature
    2) mean = Take mean of every feature
    3) concat = Concatinate every feature  
'''
def featureConsolidation(video_label_dict, video_to_patient_wise_info_mapping_dict, consolidation_type = "max", ignore_missing = False):

    ##Aggregate features
    pt_label_dict = {}

    for video, features in video_label_dict.items():
        
        if ignore_missing:
            if video not in video_to_patient_wise_info_mapping_dict:
                print(f"Ignoring missing video - {video}")
                continue

        patient_scan_name = video_to_patient_wise_info_mapping_dict[video]

        if patient_scan_name in pt_label_dict:
            consolidated_features = pt_label_dict[patient_scan_name]
        else:
            consolidated_features = {}

        for f, v in features.items():
            
            if f in consolidated_features:
                cf = consolidated_features[f]
                cf.append(v)
                consolidated_features[f] = cf
            else:
                consolidated_features[f] = [v]            

        pt_label_dict[patient_scan_name] = consolidated_features

    
    ##Combine features 
    consolidated_pt_label_dict = {}
    for patient_scan_name, pt_features in pt_label_dict.items():
        
        consolidated_pt_features = {}
        for feature_name, feature in pt_features.items():
            
            if feature_name == "unusual_findings":
                continue

            feature = np.array(feature)

            if consolidation_type == "max":
                com_feature = feature.max(axis = 0)

                assert com_feature.shape[0] == feature.shape[1], "Error! Feature consolidation done incorrectly."
            elif consolidation_type == "mean":
                com_feature = feature.mean(axis = 0)

                assert com_feature.shape[0] == feature.shape[1], "Error! Feature consolidation done incorrectly."
            elif consolidation_type == "concat":
                com_feature = np.hstack(feature)

                assert com_feature.shape[0] == feature.shape[0] * feature.shape[1], "Error! Feature consolidation done incorrectly."
            else:
                raise Exception(f"Error! Unsupported feature consoldiation type = {consolidation_type}")
            
            consolidated_pt_features[feature_name] = com_feature

        consolidated_pt_label_dict[patient_scan_name] = consolidated_pt_features
    
    assert len(consolidated_pt_label_dict.keys()) == len(pt_label_dict.keys()), "Error! Missing some consolidated features."

    return consolidated_pt_label_dict



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
                # "$X_{39}$": "d0", "$X_{40}$": "d1", "$X_{41}$": "d2", "$X_{42}$": "d3", "$X_{43}$": "d4", "$X_{44}$": "d5", "$X_{45}$": "d6" #Disease classes
            }


'''
Disease mapping dict
'''
disease_mapping_dict = {0: 0, 1: 1, 2: 5, 3: 4, 4: 5, 5: 5, 6: 2, 7: 2, 8: 3, 9: 6, 10: 5, 11: 5}

# aa = patient_data[:, patient_data_header.index('Diseases_Category')]
# bb = [disease_mapping_dict[i] for i in aa if not np.isnan(i)]
# np.unique(bb)

def main():

    labels_path = "/home/grg/Research/ENCO/datasets/user_label_ggare_2.json"

    label_dict = utils.readJson(labels_path)

    model_train_labels_path = "/home/grg/Research/ENCO/datasets/user_label_tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_C_train.json"

    model_test_labels_path = "/home/grg/Research/ENCO/datasets/user_label_tsm_LSU-Large-Dataset_BioM_Savg_EquiTS_Crop_T2_C_test_updated.json"

    model_train_labels = utils.readJson(model_train_labels_path)
    model_test_labels = utils.readJson(model_test_labels_path)

    excel_file_path = "/data1/datasets/LSU-LargeV2-Dataset/LSU_Large_Dataset_videos (final).xlsx"

    patient_data_header, patient_data = getPatientData(excel_file_path)

    # #Create a disease label dict 
    # disease_labels_dict = {}
    # for video, disease_id in zip(patient_data[:, patient_data_header.index('Video File Label (Pt # - file#)')], patient_data[:, patient_data_header.index('Diseases_Category')]):

    #     if isinstance(video, str) and not np.isnan(disease_id):
    #         video_name = f"{video.split('-')[0]}/{video}.avi"
            
    #         disease_id = disease_mapping_dict[disease_id]

    #         disease_label = [0,0,0,0,0,0,0]
    #         disease_label[disease_id] = 1

    #         disease_labels_dict[video_name] = disease_label



    patient_wise_info_dict_path = "/data1/datasets/LSU-LargeV2-Dataset/patient_wise_info_dict.json"
    patient_wise_info_dict = utils.readJson(patient_wise_info_dict_path)


    UsePerPatientFeatures = True
    consolidation_type = "max" #"max", "mean", "concat"

    if UsePerPatientFeatures:

        video_to_patient_wise_info_mapping_dict = {}
        
        consolidated_patient_info_dict = {}
        pt_SF_category_dict = {}
        pt_Disease_category_dict = {}
        pt_onehot_Disease_category_dict = {}

        for patient, scan_features in patient_wise_info_dict.items():

            for scan_date, features in scan_features.items():

                patient_scan_name = f"{patient}/{scan_date}" 

                videos = features['videos']

                for video in videos:
                    video_name = f"{video.split('-')[0]}/{video}.avi"
                    video_to_patient_wise_info_mapping_dict[video_name] = patient_scan_name

                consolidated_patient_info_dict[patient_scan_name] = {'sf_cat': features['sf_cat'], 'disease_cat': features['disease_cat']}
                
                pt_SF_category_dict[patient_scan_name] = features['sf_cat']

                disease_id = features['disease_cat']

                if not np.isnan(disease_id):
                    
                    disease_id = disease_mapping_dict[disease_id]

                    disease_label = [0,0,0,0,0,0,0]
                    disease_label[disease_id] = 1

                    pt_Disease_category_dict[patient_scan_name] = disease_id
                    pt_onehot_Disease_category_dict[patient_scan_name] = disease_label
    
        #Update exisitng video-wise dict with patient-wise dict
        SF_category_dict = pt_SF_category_dict
        disease_labels_dict = pt_onehot_Disease_category_dict


        # label_dict = featureConsolidation(label_dict, video_to_patient_wise_info_mapping_dict, consolidation_type, ignore_missing = True)
        model_train_labels = featureConsolidation(model_train_labels, video_to_patient_wise_info_mapping_dict, consolidation_type, ignore_missing = True)
        model_test_labels = featureConsolidation(model_test_labels, video_to_patient_wise_info_mapping_dict, consolidation_type, ignore_missing = True)




    ## Read Data folds 
    data_split_path = "/home/grg/Research/ENCO/datasets/dataset_split_equi_class_R1.json"
    data_split_dict = utils.readJson(data_split_path)

    train_folds = ['A', 'B']
    # train_folds = ['A', 'B', 'C']
    test_folds = ['D']
    val_folds = ['C']

    train_videos = []
    [train_videos.extend(data_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in train_folds]
    
    test_videos = []
    [test_videos.extend(data_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in test_folds]

    val_videos = []
    [val_videos.extend(data_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in val_folds]


    #Map individual video to patient-scan_dates 
    if UsePerPatientFeatures:

        train_videos = [video_to_patient_wise_info_mapping_dict[v] for v in train_videos]
        train_videos = np.unique(train_videos).tolist()

        test_videos = [video_to_patient_wise_info_mapping_dict[v] for v in test_videos]
        test_videos = np.unique(test_videos).tolist()

        val_videos = [video_to_patient_wise_info_mapping_dict[v] for v in val_videos]
        val_videos = np.unique(val_videos).tolist()



    features = ['alines', 'blines', 'blines_origin', 'pleural_thickness', 'pleural_location', 'pleural_indent', 'pleural_break', 'consolidation', 'effusion', ]

    # features = features + ['lung-severity']

    # biomarkers = []
    # for video, label in label_dict.items():
    #     v_bio = [label[f] for f in features]

    #     v_bio.append(disease_labels_dict[video])
    #     v_bio = np.hstack(v_bio)
    #     biomarkers.append(v_bio)

    
    upsampleTrainFeatures = True
    
    #Upsample train set for class balancing
    if upsampleTrainFeatures:

        train_disease_labels = []
        for video in train_videos:
            train_disease_labels.append(pt_Disease_category_dict[video])
        
        train_videos = np.array(train_videos)
        train_disease_labels = np.array(train_disease_labels)

        upsam_train_disease_labels, upsam_train_videos, upsample_label_indices = upsampleFeatures(labels = train_disease_labels, features = train_videos) 

        train_videos = upsam_train_videos.tolist()


    train_biomarkers = []
    for video in train_videos:
        label = model_train_labels[video]
        v_bio = [label[f] for f in features]

        v_bio.append(disease_labels_dict[video])
        v_bio = np.hstack(v_bio)

        train_biomarkers.append(v_bio)


    test_biomarkers = []
    for video in test_videos:
        label = model_test_labels[video]
        v_bio = [label[f] for f in features]

        v_bio.append(disease_labels_dict[video])
        v_bio = np.hstack(v_bio)

        test_biomarkers.append(v_bio)


    # val_biomarkers = []
    # for video in val_videos:
    #     label = model_val_labels[video]
    #     v_bio = [label[f] for f in features]

    #     v_bio.append(disease_labels_dict[video])
    #     v_bio = np.hstack(v_bio)

    #     val_biomarkers.append(v_bio)


    # biomarkers = np.array(biomarkers, dtype=np.uint8)
    # np.save("data1-disease.npy", biomarkers)

    # adj_matrix = np.zeros((biomarkers.shape[1], biomarkers.shape[1]), dtype=np.uint8)
    # np.save("DAG1-disease.npy", adj_matrix)

    # data_int = np.zeros((biomarkers.shape[1], 1, biomarkers.shape[1]), dtype=np.uint8)

    #Save as npz
    # np.savez("Biomarkers.npz", data_obs = biomarkers, data_int = np.zeros(1), adj_matrix = np.zeros(1))
    # np.savez("Biomarkers.npz", data_obs = biomarkers, adj_matrix = np.zeros((biomarkers.shape[1], biomarkers.shape[1])))
    # np.savez("Biomarkers.npz", 
    #             data_obs = biomarkers, 
    #             data_int = np.zeros((biomarkers.shape[1], biomarkers.shape[0], biomarkers.shape[1]), dtype=np.uint8), 
    #             adj_matrix = np.zeros((biomarkers.shape[1], biomarkers.shape[1]), dtype=np.uint8))
    # np.savez("BiomarkersGT.npz", data_obs = biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    # np.savez("DiseaseBiomarkersGTabc.npz", data_obs = biomarkers, data_int = data_int, adj_matrix = adj_matrix)


    # # train_biomarkers = np.array(train_biomarkers, dtype=np.float64)
    # train_biomarkers = np.array(train_biomarkers, dtype=np.float32)
    # np.save(f"datasets/disease_modeldata2trainUpsampledPatient-{consolidation_type}.npy", train_biomarkers)

    # # test_biomarkers = np.array(test_biomarkers, dtype=np.float64)
    # test_biomarkers = np.array(test_biomarkers, dtype=np.float32)
    # np.save(f"datasets/disease_modeldata3testUpsampledPatient-{consolidation_type}.npy", test_biomarkers)


    ##Set data type


    classification_dataset = f"ModelBiomarkers-{consolidation_type}"
    vars_list = list(mapping.values())
    class_list = [f"d{c}" for c in range(7)] #Disease-classes

    features_list = vars_list + class_list

    feature_type = {}
    for var in vars_list:
        feature_type[var] = 'continous'
    
    for cls in class_list:
        # feature_type[cls] = 'binary'
        feature_type[cls] = 'binary-class'

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

    train_biomarkers = np.array(train_biomarkers, dtype = data_type)
    test_biomarkers = np.array(test_biomarkers, dtype = data_type)
    # val_biomarkers = val_biomarkers.astype(data_type)
    
    adj_matrix = np.zeros((test_biomarkers.shape[1], test_biomarkers.shape[1]), dtype=np.uint8)

    # data_int = np.zeros((test_biomarkers.shape[1], 1, test_biomarkers.shape[1]), dtype=np.float64)
    data_int = np.zeros((test_biomarkers.shape[1], 1, test_biomarkers.shape[1]), dtype=data_type)

    # np.savez("DiseaseBiomarkersGTTrain.npz", data_obs = train_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    # np.savez("DiseaseBiomarkersGTabcTrain.npz", data_obs = train_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    # np.savez(f"DiseaseBiomarkersGTabTrainPatient-{consolidation_type}.npz", data_obs = train_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    np.savez(f"datasets/{classification_dataset}_TrainUpsampledPatient.npz", data_obs = train_biomarkers, data_int = data_int, adj_matrix = adj_matrix,
                vars_list = vars_list, class_list = class_list, feature_type = feature_type)

    # np.savez("DiseaseBiomarkersGTTest.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    # np.savez("DiseaseBiomarkersGTabcTest.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    # np.savez(f"DiseaseBiomarkersGTabTestPatient-{consolidation_type}.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    np.savez(f"datasets/{classification_dataset}_TestUpsampledPatient.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix,
                vars_list = vars_list, class_list = class_list, feature_type = feature_type)

    # np.savez(f"datasets/{classification_dataset}_ValUpsampledPatient.npz", data_obs = val_biomarkers, data_int = data_int, adj_matrix = adj_matrix,
    #             vars_list = vars_list, class_list = class_list, feature_type = feature_type)
    
    #TODO-GRG: Temporarily using test set as val set. Must change!
    np.savez(f"datasets/{classification_dataset}_ValUpsampledPatient.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix,
                vars_list = vars_list, class_list = class_list, feature_type = feature_type)
    


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
