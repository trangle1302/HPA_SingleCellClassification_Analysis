import pandas as pd
import io
import json
#!pip install pandas plotnine
from plotnine import *
import os

def read_from_json(json_file_path):
    """Function to read json file (annotation file)
    """
    with io.open(json_file_path, "r", encoding="utf-8-sig") as myfile:
        data = json.load(myfile)
    return data

def idx_to_labels(idx_list, all_locations):
    labels = []
    for idx in idx_list:
        labels += [k for k, v in all_locations.items() if v == int(float(idx))]
    return labels

def get_location_counts(label_list, all_locations):
    label_counts = dict.fromkeys(all_locations, 0)
    for l in label_list:
        if str(l) == "Discard":
            continue
        else:
            locations = idx_to_labels(str(l).split("|"), all_locations)
            for loc in locations:
                label_counts[loc] += 1
    return label_counts
    
save_dir = "/home/trangle/HPA_SingleCellClassification"
df = pd.read_excel(os.path.join(save_dir, "Kaggle_AP.xlsx"), sheet_name="Sheet1")
mapping = dict({
    "Nucleoplasm": 0,
    "Nuc.Membrane": 1,
    "Nucleoli": 2,
    "Nuc.Fib.C": 3,
    "Nuc.Speckles": 4,
    "Nuc.Bodies": 5,
    "ER": 6,
    "Golgi": 7,
    "Int.Fil": 8,
    "Actin.Fil": 9,
    "Microtubules": 10,
    "M.Spindle": 11,
    "Centrosome": 12,
    "Pl.Membrane": 13,
    "Mitochondria": 14,
    "Aggresome": 15,
    "Cytosol": 16,
    "Ves.Punctate": 17,
    "Negative" : 18
})
reverse_mapping = {str(v):k for k,v in mapping.items()}
df["ShortLabelName"] = [reverse_mapping[str(l)] for l in df.Class]

COLORS = [
    '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
    '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
    '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
    '#ff5722', '#795548', '#9e9e9e', '#607d8b', '#dddddd',
    '#212121', '#ff9e80', '#ff6d00', '#ffff00', '#76ff03',
    '#00e676', '#64ffda', '#18ffff',
]


### AP plot
# Plotting order = mean of each class
order = df.groupby(["ShortLabelName"]).mean().AP.sort_values(ascending=True).index
df.ShortLabelName = pd.Categorical(df.ShortLabelName, 
                             ordered=True,
                             categories=order)
colorvector = [COLORS[mapping[n]] for n in order]
g = ggplot(df, aes(x="ShortLabelName", y="AP", fill="ShortLabelName"))
g = g + geom_violin() + theme_classic() + theme(axis_text_x = element_text(angle=90, hjust=1)) + geom_point()+ scale_fill_manual(values=list(colorvector))
g.save(os.path.join(save_dir, "plots","AP_per_class2.png"), dpi=600)


### AP vs sample counts
gt_labels_pub = pd.read_csv("/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/publictest/labels_publictest.csv")
gt_labels_pri = pd.read_csv("/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/privatetest/labels_privatetest.csv")
gt_labels = pd.concat([gt_labels_pub, gt_labels_pri])
gt_labels["Image_ID"] = [f.split("_")[0] for f in gt_labels.ID]
gt_labels["Cell_ID"] = [f.split("_")[1] for f in gt_labels.ID]
label_counts = get_location_counts(list(gt_labels.Label), mapping)
df_ = df.groupby(["ShortLabelName"]).mean()
df_["ShortLabelName"] = df_.index
df_["GT_cell_count"] = [label_counts[l] for l in df_.index]
ggplot(df_, aes(x="ShortLabelName", y ="GT_cell_count", fill="ShortLabelName")) + geom_bar(stat = 'identity')  + theme(axis_text_x = element_text(angle=90, hjust=1))

p = ggplot(df_, aes(x="AP", y ="GT_cell_count", fill="ShortLabelName")) + geom_point() + theme_classic() + xlim(0.2, 0.8) + ylim(0, 12000) + geom_text(aes(label="ShortLabelName"), size=7, angle=20, ha='left', va='bottom') 
p.save(os.path.join(save_dir, "plots","Cellcounts_vs_AP.jpg"), dpi=600)
