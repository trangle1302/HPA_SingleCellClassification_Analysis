import pandas as pd
import io
import json
#!pip install pandas plotnine
from plotnine import *
import os
import seaborn as sns
from tqdm import tqdm

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

def merge_labels(location_names):
    loc_merged = {
        "Vesicles and punctate cytosolic patterns": [
            "Vesicles",
            "Peroxisomes",
            "Endosomes",
            "Lysosomes",
            "Lipid droplets",
            "Cytoplasmic bodies",
        ],
        "Centrosome": ["Centrosome", "Centriolar satellite"],
        "Plasma membrane": ["Plasma membrane", "Cell Junctions"],
    }
    loc_map = {
        "Vesicles": "Vesicles and punctate cytosolic patterns",
        "Peroxisomes": "Vesicles and punctate cytosolic patterns",
        "Endosomes": "Vesicles and punctate cytosolic patterns",
        "Lysosomes": "Vesicles and punctate cytosolic patterns",
        "Lipid droplets": "Vesicles and punctate cytosolic patterns",
        "Cytoplasmic bodies": "Vesicles and punctate cytosolic patterns",
        "Centrosome": "Centrosome",
        "Centriolar satellite": "Centrosome",
        "Plasma membrane": "Plasma membrane",
        "Cell Junctions": "Plasma membrane",
        "Focal adhesion sites": "Actin filaments",
    }
    location_names_return = location_names.copy()
    for i, loc in enumerate(location_names_return):
        if loc in list(loc_map.keys()):
            location_names_return[i] = loc_map[loc]
    return location_names_return

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

COLORS = [
    '#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5',
    '#2196f3', '#03a9f4', '#00bcd4', '#009688', '#4caf50',
    '#8bc34a', '#cddc39', '#ffeb3b', '#ffc107', '#ff9800',
    '#ff5722', '#795548', '#9e9e9e', '#607d8b', '#dddddd',
    '#212121', '#ff9e80', '#ff6d00', '#ffff00', '#76ff03',
    '#00e676', '#64ffda', '#18ffff',
]

    
save_dir = "/home/trangle/HPA_SingleCellClassification"
df = pd.read_excel(os.path.join(save_dir, "Kaggle_AP.xlsx"), sheet_name="Sheet1")
df["ShortLabelName"] = [reverse_mapping[str(l)] for l in df.Class]

# Mean and Std of each classes for top 50 teams
aggregated_performance = df.groupby('LabelName').agg({
    'AP_public': ['mean','std'],
    'AP_private': ['mean','std']
    }).reset_index()
aggregated_performance.to_csv(f'{save_dir}/Kaggle_AP_mean_std.csv', index=False)

### Label counts of train and test sets
label_df = pd.read_csv('/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/kaggle_master_training_reindex.csv')
label_df = label_df[~label_df.locations_reindex.isna()]
label_counts_train = get_location_counts(label_df.locations_reindex.values.tolist(), mapping)

label_df = pd.read_csv('/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/IF-image_reindex.csv')
label_df = label_df[label_df.latest_version==19]
label_df = label_df[~label_df.locations_reindex.isna()]
label_counts_pHPA = get_location_counts(label_df.locations_reindex.values.tolist(), mapping)
LABEL_NAMES_REV = dict([(v,k) for k,v in LABEL_NAMES.items()])
ifimage = pd.read_csv('/home/trangle/Downloads/IF-image.csv')
ifimage = ifimage[ifimage.latest_version==20]
ifimage['Merged_label'] = ''
ifimage['LabelIndex'] = ''
idx_nan = []
for i,row in tqdm(ifimage.iterrows(), total=ifimage.shape[0]):
    try:
        img_locations = merge_labels(row.locations.split(','))
        ifimage.loc[i,'Merged_label'] = ",".join(img_locations)
        img_locations = [org for org in img_locations if org in LABEL_NAMES_REV.keys()]
        ifimage.loc[i,'LabelIndex'] = "|".join([str(LABEL_NAMES_REV[org]) for org in img_locations])
    except:
        if row.unspecific == 0: #if there's no label and not unspecific, then it's negative
            ifimage.loc[i,'Merged_label'] = "Negative" 
            ifimage.loc[i,'LabelIndex'] = "18"
        idx_nan += [i]
tmp = ifimage[ifimage.LabelIndex!='']
labelcount_2021 = get_location_counts(tmp.LabelIndex.values.tolist(), mapping)

label_df = pd.read_csv('/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/privatetest/labels_privatetest.csv')
label_counts_private = get_location_counts(label_df.Label.values.tolist(), mapping)
label_df = pd.read_csv('/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/publictest/labels_publictest.csv')
label_counts_public = get_location_counts(label_df.Label.values.tolist(), mapping)

# Merge all to label_counts
label_counts = {'train': label_counts_train, 'pHPA':label_counts_pHPA, 'publictest':label_counts_public, 'privatetest':label_counts_private}

plot_order = [k for k,v in sorted(label_counts_private.items(), key=lambda item: item[1], reverse=True)]


label_counts =pd.DataFrame.from_dict(label_counts)
label_counts["ShortLabelName"] = label_counts.index
label_counts.to_csv(f'{save_dir}/Label_distribution.csv', index=False)
label_counts = pd.melt(label_counts, id_vars=["ShortLabelName"])
label_counts.ShortLabelName = pd.Categorical(label_counts.ShortLabelName, 
                             ordered=True,
                             categories=plot_order)
label_counts["unit"] = "cell"
label_counts["unit"][label_counts.variable=="train"] = "image" 
label_counts["unit"][label_counts.variable=="pHPA"] = "image" 

g = ggplot(label_counts, aes(x="ShortLabelName", y="value", fill="variable"))
g = g + geom_bar(stat = "identity",position="stack") + facet_grid("unit~.") + theme_classic() + theme(axis_text_x = element_text(angle=90, hjust=1)) + scale_fill_manual(values=["blue","lightgreen","green","orange",])
g.save(os.path.join(save_dir, "plots","Label_distribution.png"), dpi=600)


g = ggplot(label_counts[label_counts.unit == "cell"], aes(x="ShortLabelName", y="value", fill="variable"))
g = g + geom_bar(stat = "identity",position="stack") + theme_classic() + theme(axis_text_x = element_text(angle=90, hjust=1)) + scale_fill_manual(values=["lightgreen","green",])
g.save(os.path.join(save_dir, "plots","Label_distribution_cell.png"), dpi=600)

g = ggplot(label_counts[label_counts.unit == "image"], aes(x="ShortLabelName", y="value", fill="variable"))
g = g + geom_bar(stat = "identity",position="stack") +  theme_classic() + theme(axis_text_x = element_text(angle=90, hjust=1)) + scale_fill_manual(values=["blue","orange",])
g.save(os.path.join(save_dir, "plots","Label_distribution_image.png"), dpi=600)

### AP plot
# Plotting order = mean of each class
#order = df.groupby(["ShortLabelName"]).mean().AP_private.sort_values(ascending=True).index
df.ShortLabelName = pd.Categorical(df.ShortLabelName, 
                             ordered=True,
                             categories=plot_order)

colorvector = [COLORS[mapping[n]] for n in plot_order]
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
