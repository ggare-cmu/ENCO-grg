from termios import VDISCARD
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
Disease mapping dict
'''
disease_mapping_dict = {0: 0, 1: 1, 2: 5, 3: 4, 4: 5, 5: 5, 6: 2, 7: 2, 8: 3, 9: 6, 10: 5, 11: 5}

# aa = patient_data[:, patient_data_header.index('Diseases_Category')]
# bb = [disease_mapping_dict[i] for i in aa if not np.isnan(i)]
# np.unique(bb)

def main():

    labels_path = "/home/grg/Research/ENCO/user_label_ggare_2.json"

    label_dict = utils.readJson(labels_path)

    excel_file_path = "/data1/datasets/LSU-LargeV2-Dataset/LSU_Large_Dataset_videos (final).xlsx"

    patient_data_header, patient_data = getPatientData(excel_file_path)

    #Create a disease label dict 
    disease_labels_dict = {}
    for video, disease_id in zip(patient_data[:, patient_data_header.index('Video File Label (Pt # - file#)')], patient_data[:, patient_data_header.index('Diseases_Category')]):

        if isinstance(video, str) and not np.isnan(disease_id):
            video_name = f"{video.split('-')[0]}/{video}.avi"
            
            disease_id = disease_mapping_dict[disease_id]

            disease_label = [0,0,0,0,0,0,0]
            disease_label[disease_id] = 1

            disease_labels_dict[video_name] = disease_label




    ## Read Data folds 
    data_split_path = "/home/grg/Research/ENCO/dataset_split_equi_class_R1.json"
    data_split_dict = utils.readJson(data_split_path)

    train_folds = ['A', 'B']
    # train_folds = ['A', 'B', 'C']
    test_folds = ['D']

    train_videos = []
    [train_videos.extend(data_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in train_folds]
    
    test_videos = []
    [test_videos.extend(data_split_dict[f"fold{fold}"][f"{fold}_pt_videos"]) for fold in test_folds]


    features = ['alines', 'blines', 'blines_origin', 'pleural_thickness', 'pleural_location', 'pleural_indent', 'pleural_break', 'consolidation', 'effusion', ]

    # features = features + ['lung-severity']

    # biomarkers = []
    # for video, label in label_dict.items():
    #     v_bio = [label[f] for f in features]

    #     v_bio.append(disease_labels_dict[video])
    #     v_bio = np.hstack(v_bio)
    #     biomarkers.append(v_bio)

    
    train_biomarkers = []
    for video in train_videos:
        label = label_dict[video]
        v_bio = [label[f] for f in features]

        v_bio.append(disease_labels_dict[video])
        v_bio = np.hstack(v_bio)

        train_biomarkers.append(v_bio)


    test_biomarkers = []
    for video in test_videos:
        label = label_dict[video]
        v_bio = [label[f] for f in features]

        v_bio.append(disease_labels_dict[video])
        v_bio = np.hstack(v_bio)

        test_biomarkers.append(v_bio)


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


    train_biomarkers = np.array(train_biomarkers, dtype=np.uint8)
    np.save("disease_data2train.npy", train_biomarkers)

    test_biomarkers = np.array(test_biomarkers, dtype=np.uint8)
    np.save("disease_data3test.npy", test_biomarkers)


    adj_matrix = np.zeros((test_biomarkers.shape[1], test_biomarkers.shape[1]), dtype=np.uint8)

    data_int = np.zeros((test_biomarkers.shape[1], 1, test_biomarkers.shape[1]), dtype=np.uint8)


    # np.savez("DiseaseBiomarkersGTTrain.npz", data_obs = train_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    # np.savez("DiseaseBiomarkersGTabcTrain.npz", data_obs = train_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    np.savez("DiseaseBiomarkersGTabTrain.npz", data_obs = train_biomarkers, data_int = data_int, adj_matrix = adj_matrix)

    # np.savez("DiseaseBiomarkersGTTest.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    # np.savez("DiseaseBiomarkersGTabcTest.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix)
    np.savez("DiseaseBiomarkersGTabTest.npz", data_obs = test_biomarkers, data_int = data_int, adj_matrix = adj_matrix)

    


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
