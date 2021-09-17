import os
import pandas as pd
from sklearn.metrics import f1_score

bestfitting_match = pd.read_csv("/data/kaggle-dataset/mAPscoring/bestfitting/IOU_p_merged.csv")

def conf_to_label(pred_str, conf_threshold = 0.5):
    d = eval(pred_str)
    labels = None
    for k, v in d.items():
        if float(v) > conf_threshold:
            if labels == None:
                labels = k
            else:
                labels = "|".join([labels, k])
    return labels


bestfitting_match["Predicted_cell_label_formatted"] = "" 
for i, row in bestfitting_match.iterrows():
    pred_str = row.Predicted_cell_label
    #print(pred_str)
    if pred_str != 'None':
        formatted_str = conf_to_label(pred_str, conf_threshold=0.2)
        if formatted_str != None:
            bestfitting_match.loc[i,"Predicted_cell_label_formatted"] = formatted_str

bestfitting_match_df = bestfitting_match[bestfitting_match.GT_cell_label!='None']
n_GT = bestfitting_match_df.GT_cell_label.value_counts().sum()
n_matched = sum(bestfitting_match_df.Predicted_cell_label!= "None")
print(f"Predicted {n_matched}/{n_GT} cells")

