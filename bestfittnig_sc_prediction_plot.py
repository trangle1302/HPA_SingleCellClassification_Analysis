import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "/home/trangle/HPA_SingleCellClassification/hpa2021_0902"
IMAGE_DIR = "/data/kaggle-dataset/PUBLICHPA"
SAVE_DIR = "/home/trangle/HPA_SingleCellClassification/plots"

LABEL_TO_NAME = dict({
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
  18: 'Negative',
})
LABEL_TO_ALIAS = dict({
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
  18: 'Negative',
})
ALIAS_TO_LABEL = {str(v):k for k,v in LABEL_TO_ALIAS.items()}

df = pd.read_csv(os.path.join(BASE_DIR,"result/d0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds/fold0/epoch_12.00_ema/cell_result_train_cell_v1.csv"))
csv = pd.read_csv(os.path.join(IMAGE_DIR,"mask/train.csv"))


for (k,v) in ALIAS_TO_LABEL.items():
    plt.hist(df[k], bins=1000, alpha = 0.3)
    plt.xlabel("Class confidence score")
    plt.ylabel("Cell counts")
    plt.title(LABEL_TO_NAME[v])
    plt.savefig(os.path.join(SAVE_DIR,f"confidence_by_bestfittinghpasc_for_PUBLICHPA_{k}.png"))
    
    
THRESHOLD = 0.5

