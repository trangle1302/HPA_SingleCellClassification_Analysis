# Prediction of single-label overlap % single-label in HPA

# Prediction of multi-label overlap % multi-locations in HPA

# Single cell labels 

import ast
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve


NUM_CLASSES = 20
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
  18: 'Negative'
}

matched_cells_path = '/data/kaggle-dataset/mAPscoring/bestfitting/IOU_p_merged_inv3.csv'
pred = pd.read_csv(matched_cells_path)
pred.GT_cell_label = [f.replace("['","").replace("']","") for f in pred.GT_cell_label]
pred = pred[pred.GT_cell_label!='Discard']
pred = pred[pred.Predicted_cell_label!='None']
thres_df = []
for LABEL, ALIAS in LABEL_TO_ALIAS.items():
  y_true = [1 if str(LABEL) in f.split('|') else 0 for f in pred.GT_cell_label ]
  print(f'Number of GT for {ALIAS}: {sum(y_true)}/{len(y_true)}')
  y_score = [np.float(ast.literal_eval(f)[str(LABEL)]) for f in pred.Predicted_cell_label]
  precision, recall, thresholds = precision_recall_curve(y_true, y_score)
  denominator = (recall+precision)
  denominator[denominator==0] = 1e-10 #subsitute 0 with very small number to be divisible in the next step
  f1_scores = 2*recall*precision/denominator
  print('Best threshold: ', thresholds[np.argmax(f1_scores)])
  print('Best F1-Score: ', np.max(f1_scores))
  line = [LABEL,ALIAS,sum(y_true),thresholds[np.argmax(f1_scores)],np.max(f1_scores)]
  thres_df.append(line)
thres_df = pd.DataFrame(thres_df)
thres_df.columns = ['Label','Alias','n_gt','best_threshold','best_f1']
print(thres_df)
print(thres_df.best_f1.mean())
thres_df.to_csv('/data/kaggle-dataset/mAPscoring/bestfitting/thres_inv3.csv', index=False)