import pandas as pd
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from numpy.matrixlib.defmatrix import matrix
import numpy as np
import os
import gc
from sklearn.metrics import f1_score
#import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
#%matplotlib inline 
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
sns.set(style='white', color_codes=True)

osp = os.path
ope = os.path.exists
opj = os.path.join

#%%
FEATURE_DIR    = '/home/trangle/HPA_SingleCellClassification/team1_bestfitting/features'
DATA_DIR       = '/data/kaggle-dataset/PUBLICHPA'
DATASET        = 'train'
MODEL_NAME     = 'd0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds'

ID             = 'ID'
LABEL          = 'Label'
WIDTH          = 'ImageWidth'
HEIGHT         = 'ImageHeight'
TARGET         = 'PredictionString'
CELL_LINE      = 'Cellline'
ANTIBODY       = 'Antibody'
ANTIBODY_LABEL = 'AntibodyLabel'

NUM_CLASSES = 20
ML_NUM_CLASSES = 20000

NEGATIVE = 18

LABEL_NAMES = {
  0: 'Nucleoplasm',
  1: 'Nuclear membrane',
  2: 'Nucleoli',
  3: 'Nucleoli fibrillar center',
  4: 'Nuclear speckles',
  5: 'Nuclear bodies',
  6: 'Endoplasmic reticulum',
  7: 'Golgi apparatus',
  8: 'Intermediate filaments',
  9: 'Actin filaments',
  10: 'Microtubules',
  11: 'Mitotic spindle',
  12: 'Centrosome',
  13: 'Plasma membrane',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'Vesicles and punctate cytosolic patterns',
  NEGATIVE: 'Negative',
  19:'Multi-Location',
}
LABEL_TO_ALIAS = {
  0: 'Nucleoplasm',
  1: 'NuclearM',
  2: 'Nucleoli',
  3: 'NucleoliFC',
  4: 'NuclearS',
  5: 'NuclearB',
  6: 'EndoplasmicR',
  7: 'GolgiA',
  8: 'IntermediateF',
  9: 'ActinF',
  10: 'Microtubules',
  11: 'MitoticS',
  12: 'Centrosome',
  13: 'PlasmaM',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'VesiclesPCP',
  NEGATIVE: 'Negative',
  19:'Multi-Location',
}
LABEL_NAME_LIST = [LABEL_NAMES[i] for i in range(NUM_CLASSES-1)]
LABEL_ALIASE_LIST = [LABEL_TO_ALIAS[i] for i in range(NUM_CLASSES-1)]

COLORS = [
    '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
    '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
    '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
    '#ff5722', '#795548', '#9e9e9e', '#607d8b', '#dddddd',
    '#212121', '#ff9e80', '#ff6d00', '#ffff00', '#76ff03',
    '#00e676', '#64ffda', '#18ffff',
]
MERGE_TYPE = 'quarterthreshold' # 'addSC'


LABEL_NAMES_MERGED = {
  'Nucleoplasm':'Nucleoplasm',
  'Nuclear membrane':'Nuclear membrane',
  'Nucleoli':'Nucleoli',
  'Nucleoli fibrillar center':'Nucleoli fibrillar center',
  'Nuclear speckles':'Nuclear speckles',
  'Nuclear bodies':'Nuclear bodies',
  'Endoplasmic reticulum':'Endoplasmic reticulum',
  'Golgi apparatus':'Golgi apparatus',
  'Intermediate filaments':'Intermediate filaments',
  'Actin filaments':'Actin filaments',
  'Focal adhesion sites':'Actin filaments',
  'Microtubules':'Microtubules',
  'Mitotic spindle':'Mitotic spindle',
  'Centrosome':'Centrosome',
  'Centriolar satellite':'Centrosome',
  'Plasma membrane':'Plasma membrane',
  'Cell Junctions':'Plasma membrane',
  'Mitochondria':'Mitochondria',
  'Aggresome':'Aggresome',
  'Cytosol':'Cytosol',
  'Vesicles':'Vesicles and punctate cytosolic patterns',
  'Peroxisomes':'Vesicles and punctate cytosolic patterns',
  'Endosomes':'Vesicles and punctate cytosolic patterns',
  'Lysosomes':'Vesicles and punctate cytosolic patterns',
  'Lipid droplets':'Vesicles and punctate cytosolic patterns',
  'Cytoplasmic bodies':'Vesicles and punctate cytosolic patterns'
}

def roundToNearest(inputNumber, base=0.25):
    return base*np.round(inputNumber/base)

def prepare_meta_publicHPA():
    cells_publicHPA_path = f'{DATA_DIR}/inputs/cells_publicHPA.csv'
    print(f'Loading {cells_publicHPA_path}')
    ifimages_v20_ = pd.read_csv(cells_publicHPA_path)    

    ### Merge HPA image-level labels and predicted SC levels from bestfitting best single model
    # Do some rules like bestfitting to combine image-level labels and predicted SC labels?
    prediction = pd.read_csv(f'{FEATURE_DIR.replace("features","result")}/{MODEL_NAME}/fold0/epoch_12.00_ema/cell_result_test_cell_v1.csv')
    prediction["cellmask"] = prediction["mask"]
    prediction = prediction.drop(columns=["mask"])    
    tmp = prediction.merge(ifimages_v20_, how='inner', on=['ID', 'cellmask','maskid'])
    
    if MERGE_TYPE == 'quarterthreshold':
        il_labels = tmp[[l+'_y' for l in LABEL_ALIASE_LIST]].values
        sc_labels = tmp[[l+'_x' for l in LABEL_ALIASE_LIST]].values
        sc_labels = np.array([c/c.max() for c in sc_labels])
        # sc_labels = list(map(lambda row: [roundToNearest(c, 0.25) for c in row], sc_labels))
        sc_labels = [roundToNearest(c, 0.25) for c in sc_labels]
        sc_labels = pd.DataFrame(np.round(il_labels*sc_labels).astype('uint8'))
    elif MERGE_TYPE == 'addSC':
        il_labels = tmp[[l+'_y' for l in LABEL_ALIASE_LIST]].values
        sc_labels = tmp[[l+'_x' for l in LABEL_ALIASE_LIST]].values
        sc_labels = (il_labels+sc_labels)/2
        sc_labels = np.array([c/c.max() for c in sc_labels])
        sc_labels = pd.DataFrame(np.round(sc_labels).astype('uint8'))
    sc_labels.columns = LABEL_ALIASE_LIST
    
    df_c = pd.concat([tmp[["ID", "Label", "maskid", 'cellmask', 'ImageWidth', 'ImageHeight']], sc_labels], axis=1)
    labels = df_c[LABEL_ALIASE_LIST].values
    negatives_idx = np.where(labels.sum(axis=1)==0)[0]
    df_c.loc[negatives_idx, "Negative"] = 1

    single_label_idx = np.where((labels==1).sum(axis=1)==1)[0]
    single_labels = labels[single_label_idx]
    idx1 = np.where(single_labels==1)
    single_labels = [LABEL_ALIASE_LIST[i] for i in idx1[1]]
    multi_label_idx = np.where((labels==1).sum(axis=1)>1)[0]
    multi_labels = [list(LABEL_NAMES.values())[-1] for i in multi_label_idx]

    df_c['target'] = 'Negative'
    df_c.loc[single_label_idx, 'target'] = single_labels
    df_c.loc[multi_label_idx, 'target'] = multi_labels
    df_c['cellid'] = ['_'.join((r.ID, str(r.maskid))) for _,r in df_c.iterrows()]
    prediction['cellid'] = ['_'.join((r.ID, str(r.maskid))) for _,r in prediction.iterrows()]
    df_c["prob"] = 0
    for i,row in df_c.iterrows():
        label = row.target
        if label == 'Multi-Location':
            df_c.loc[i,'prob'] = 1
        else:
            #print(prediction[prediction.cellid == row.cellid][label].values)
            df_c.loc[i,'prob'] = prediction[prediction.cellid == row.cellid][label].values[0]
    df_c.target.value_counts()
    df_c.to_csv(f'{DATA_DIR}/inputs/cells_publicHPA_mergedSCprediction_{MERGE_TYPE}.csv', index=False)
    

def prepare_meta_publicHPAv21():
    prediction = pd.read_csv(f'{FEATURE_DIR.replace("features","result")}/{MODEL_NAME}/fold0/epoch_12.00_ema/cell_result_test_cell_v1.csv')
    prediction["cellmask"] = prediction["mask"]
    prediction = prediction.drop(columns=["mask"])  
    prediction['id'] = ['_'.join((r.ID, str(r.maskid))) for _,r in prediction.iterrows()]
    d = '/data/kaggle-dataset/publicHPA_umap/results/webapp'
    predictions0 = pd.read_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox.csv')
    predictions0['id'] = [f.split('_',1)[1] for f in predictions0.id]
    prediction = prediction.merge(predictions0.drop(columns=['location_code', 'locations','gene_names', 'ensembl_ids', 'atlas_name','target']), on = ['id'])
    del(predictions0)
    gc.collect()
    ifimage = pd.read_csv('/data/HPA-IF-images/IF-image.csv')
    ifimage["ID"] = [f.rsplit('/',1)[1][:-1] for f in ifimage.filename]
    ifimage = ifimage[ifimage.ID.isin(prediction.ID)]
    print(ifimage['Ab state'].value_counts())
    print('Taking only image with "IF_FINISHED"')
    ifimage = ifimage[ifimage['Ab state']=="IF_FINISHED"]
    all_labels = [l.split(',') for l in ifimage.locations if str(l)!='nan']
    all_labels = set([sl for l in all_labels for sl in l])
    rm_labels = all_labels.difference(LABEL_NAMES_MERGED.keys())
    tmp = []
    for i,r in ifimage.iterrows():
        try:
            ls = r.locations.split(',')
            ls = [l for l in ls if l not in rm_labels]
            if len(ls) > 1:
                ls = [LABEL_NAMES_MERGED[l] for l in ls]
                tmp.append(",".join(ls))
            elif len(ls) == 1:
                tmp.append(LABEL_NAMES_MERGED[ls[0]])
            else:
                tmp.append("")
        except:
            tmp.append("")
    ifimage["locations_cleanedup"] = tmp
    print('Done with locations cleanup for HPASC comp')
    ifimage = ifimage.reset_index()
    m = np.zeros((ifimage.shape[0],len(LABEL_ALIASE_LIST)))
    for i,r in ifimage.iterrows():
        if r.locations_cleanedup != "":
            ls = r.locations_cleanedup.split(',')
            for l in ls:
                m[i,LABEL_NAME_LIST.index(l)] = 1
    m[m!=1]=0
    print('Done with one-hot encoding for locations_leanedup')
    df = pd.DataFrame(m) # m= matrix of image-level labels
    df.columns = LABEL_ALIASE_LIST

    single_label_idx = np.where((m==1).sum(axis=1)==1)[0]
    single_labels = m[single_label_idx]
    idx1 = np.where(single_labels==1)
    single_labels = [LABEL_ALIASE_LIST[i] for i in idx1[1]]

    multi_label_idx = np.where((m==1).sum(axis=1)>1)[0]
    multi_labels = [list(LABEL_NAMES.values())[-1] for i in multi_label_idx]

    df['target'] = 'unknown'
    df.loc[single_label_idx, 'target'] = single_labels
    df.loc[multi_label_idx, 'target'] = multi_labels
    df['target'].value_counts()
    df['ID'] = ifimage.ID
    df = df.merge(ifimage[['ID','locations','locations_cleanedup','gene_names','ensembl_ids','atlas_name']], on=['ID'])
    print(f'Current columns: {df.columns}')
    # Merged labels in previous multiplication step, now reformat target column
    tmp = prediction.merge(df, how='inner', on=['ID'])
    print(f'Current columns: {tmp.columns}')
    print(f'Current shape: {tmp.shape}')

    # Merge il and sc
    thres_classes = pd.read_csv('/data/kaggle-dataset/mAPscoring/bestfitting/thres_inv3.csv')
    
    assert (thres_classes.Alias.values == LABEL_ALIASE_LIST).all()
    thres_vector = thres_classes.best_threshold
    il_labels = tmp[[l+'_y' for l in LABEL_ALIASE_LIST]].values
    sc_labels = tmp[[l+'_x' for l in LABEL_ALIASE_LIST]].values
    sc_labels = np.array([sc_labels.transpose()[c] > (2*thres_vector[c]) for c in range(19)]).transpose()
    sc_labels = il_labels*sc_labels 
    print(sc_labels.shape)
    sc_labels = pd.DataFrame(sc_labels.astype('uint8'))
    sc_labels.columns = LABEL_ALIASE_LIST
    # Merge the calculated sc_labels back with meta
    df_c = pd.concat([tmp[['id',"ID", 'maskid', 'locations','locations_cleanedup','gene_names','ensembl_ids','atlas_name','x','y','z','top','left','width','height',"ImageWidth"]], sc_labels], axis=1)
    df_c.to_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_metav21_lowermerge.csv', index=False)
    print(f'Wrote intermediate results in {d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_metav21.csv')
    del(il_labels,sc_labels,df, prediction)
    gc.collect()
    labels = df_c[LABEL_ALIASE_LIST].values
    negatives_idx = np.where(labels.sum(axis=1)==0)[0]
    df_c.loc[negatives_idx, "Negative"] = 1
    # Find names of single labels and multi labels
    single_label_idx = np.where((labels==1).sum(axis=1)==1)[0]
    single_labels = labels[single_label_idx]
    idx1 = np.where(single_labels==1)
    single_labels = [LABEL_ALIASE_LIST[i] for i in idx1[1]]
    multi_label_idx = np.where((labels==1).sum(axis=1)>1)[0]
    multi_labels = [list(LABEL_NAMES.values())[-1] for i in multi_label_idx]
    # Write single cell multilabel names
    multi_labels_names = [np.where(r==1)[0] for r in labels[multi_label_idx]]
    multi_labels_names = [','.join([LABEL_NAMES[i] for i in r]) for r in multi_labels_names]
    # Fill sc 'location' and target column
    df_c['location'] = 'Negative'
    df_c['target'] = 'Negative'
    df_c.loc[single_label_idx, 'location'] = single_labels
    df_c.loc[single_label_idx, 'target'] = single_labels
    df_c.loc[multi_label_idx, 'location'] = multi_labels_names
    df_c.loc[multi_label_idx, 'target'] = multi_labels
    # Add probability
    sc_labels = tmp[[l+'_x' for l in LABEL_ALIASE_LIST]].values
    print(f'sc prob shape {sc_labels.shape}, df shape {df_c.shape}')
    prob = []
    for i,row in df_c.iterrows():
        label = row.target
        if label == 'Multi-Location':
            prob.append(1)
        else:
            prob.append(sc_labels[i,LABEL_ALIASE_LIST.index(label)])
    df_c["prob"] = prob
    df_c.target.value_counts()
    print('Keeping these final columns:')
    print(df_c.columns)
    df_c.to_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_metav21_individualthresholds_il.csv', index=False)
    #df_c.to_csv(f'{d}/sl_pHPA_15_0.05_euclidean_100000_rmoutliers_ilsc_3d_bbox_metav21_potentialnewlabels.csv', index=False)


if __name__ == '__main__':
    #prepare_meta_publicHPA()
    prepare_meta_publicHPAv21()