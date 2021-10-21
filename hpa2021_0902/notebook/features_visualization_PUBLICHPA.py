import pandas as pd
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from numpy.matrixlib.defmatrix import matrix
import numpy as np
import os
from sklearn.metrics import f1_score
#import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
#%matplotlib inline 
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
sns.set(style='white', color_codes=True)


from sklearn import preprocessing
from sklearn.manifold import TSNE
import umap

osp = os.path
ope = os.path.exists
opj = os.path.join

#%%
FEATURE_DIR    = '/home/trangle/HPA_SingleCellClassification/hpa2021_0902/features'
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


def prepare_train_features():
    valid_df = pd.read_csv('/home/trangle/HPA_SingleCellClassification/hpa2021_0902/data/split/random_folds5/random_valid_cv0.csv')
    valid_df.shape

    df = pd.read_csv(f'/home/trangle/HPA_SingleCellClassification/hpa2021_0902/data/mask/{DATASET}.csv')
    df['valid'] = 0
    df.loc[df[ID].isin(valid_df[ID].values), 'valid'] = 1
    print(df.shape)
    df.head()

    # Here is the label of the cells, you can use your truth cells label.
    # The cells were labeled to 5 levels with label [1.0, 0.75, 0.5, 0.25, 0 ], 
    # this is a rule based procedure, After getting the outputs of all cells of train set from FCAN, 
    # we can give higher label value if the image probability and cell probability are high, 
    # and the cells from an image with label A were given at least 0.25 of this label A. 
    # The thresholds of the rule were not sensitive according to my experiments.
    cell_df = pd.read_csv(f'/home/trangle/HPA_SingleCellClassification/hpa2021_0902/data/inputs/cellv4b_{DATASET}.csv')
    print(cell_df.shape)
    cell_df.head()

    df = df.merge(cell_df, how='left', on=[ID, 'maskid'])
    df.shape

    labels = df[LABEL_ALIASE_LIST].values
    single_label_idx = np.where((labels==1).sum(axis=1)==1)[0]
    single_labels = labels[single_label_idx]
    idx1 = np.where(single_labels==1)
    single_labels = [LABEL_ALIASE_LIST[i] for i in idx1[1]]

    multi_label_idx = np.where((labels==1).sum(axis=1)>1)[0]
    multi_labels = [list(LABEL_NAMES.values())[-1] for i in multi_label_idx]

    df['target'] = 'unknown'
    df.loc[single_label_idx, 'target'] = single_labels
    df.loc[multi_label_idx, 'target'] = multi_labels
    df['target'].value_counts()

    # Load trainval features
    file_name = f'cell_features_{DATASET}_default_cell_v1_trainvalid.npz'
    features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name}'
    features = np.load(features_file, allow_pickle=True)['feats']
    features.shape

    # filter train features
    train_df = df[(df['target']!='unknown') & df['valid']==0]
    train_df = train_df.groupby('target').head(1000)
    train_features = features[train_df.index]
    train_features.shape
    train_features.to_csv(f'{FEATURE_DIR}/inputs/cells_publicHPA.csv', index=False)

def prepare_meta_publicHPA():
    ifimages_v20_ = pd.read_csv(f"{DATA_DIR}/inputs/train.csv")
    masks = pd.read_csv(f"{DATA_DIR}/mask/train.csv")
    ifimages_v20_ = ifimages_v20_.merge(masks, how='inner', on=['ID', 'ImageHeight', 'ImageWidth'])

    labels = ifimages_v20_[LABEL_ALIASE_LIST].values
    single_label_idx = np.where((labels==1).sum(axis=1)==1)[0]
    single_labels = labels[single_label_idx]
    idx1 = np.where(single_labels==1)
    single_labels = [LABEL_ALIASE_LIST[i] for i in idx1[1]]
    multi_label_idx = np.where((labels==1).sum(axis=1)>1)[0]
    multi_labels = [list(LABEL_NAMES.values())[-1] for i in multi_label_idx]

    ifimages_v20_['target'] = 'unknown'
    ifimages_v20_.loc[single_label_idx, 'target'] = single_labels
    ifimages_v20_.loc[multi_label_idx, 'target'] = multi_labels

    ifimages_v20_.target.value_counts()
    ifimages_v20_.to_csv(f'{DATA_DIR}/inputs/cells_publicHPA.csv', index=False)


def show_features(train_features, features_publicHPA, sub_df, show_multi=True, title=''):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean',random_state=33)
    reducer.fit(train_features)
    X = reducer.transform(features_publicHPA.tolist())

    num_classes = NUM_CLASSES if show_multi else NUM_CLASSES-1
    fig, ax = plt.subplots(figsize=(32, 16))
    for i in range(num_classes):
        label = LABEL_TO_ALIAS[i]
        idx = np.where(sub_df['target']==label)[0]
        x = X[idx, 0]
        y = X[idx, 1]
        plt.scatter(x, y, c=COLORS[i],label=LABEL_TO_ALIAS[i], s=16)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width* 0.8, box.height])
    ax.legend(loc='upper right', fontsize=24, bbox_to_anchor=(1.24, 1.01), ncol=1)
    plt.title(title, fontsize=24)
    sub_df["x"] = [idx[0] for idx in X]
    sub_df["y"] = [idx[1] for idx in X]
    sub_df["id"] = ["_".join([img,str(cell)]) for img, cell in zip(sub_df.ID, sub_df.maskid)]
    sub_df.to_csv(f"{DATA_DIR}/{title}.csv", index=False)
    return sub_df

def main():

    prepare_meta_publicHPA()
    # Load train features:
    file_name = f'cell_features_{DATASET}_default_cell_v1_trainvalid.npz'
    features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name}'
    train_features = np.load(features_file, allow_pickle=True)['feats']
    train_features.shape

    # Load publicHPA features
    file_name_publicHPA = 'cell_features_train_default_cell_v1_publicHPA.npz'
    features_file = f'{FEATURE_DIR}/{MODEL_NAME}/fold0/epoch_12.00_ema/{file_name_publicHPA}'
    features_publicHPA = np.load(features_file, allow_pickle=True)['feats']
    features_publicHPA.shape

    # Preprocess
    X = preprocessing.scale(np.vstack((train_features, features_publicHPA)))
    train_features = X[:len(train_features)]

    # Fit and transform
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean',random_state=33)
    reducer.fit(train_features)
    X = reducer.transform(features_publicHPA.tolist())
    np.savez_compressed(f"{DATA_DIR}/transformed_publicHPA", feats=X)
    #X = np.load(f"{DATA_DIR}/transformed_publicHPA", allow_pickle=True)['feats']
if __name__ == '__main__':
    main()
