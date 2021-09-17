import pandas as pd
import io
import json
#!pip install pandas plotnine
from plotnine import *

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
    

df = pd.read_excel("/home/trangle/HPA_SingleCellClassification/Kaggle_AP.xlsx", sheet_name="Sheet1")
#mapping = pd.read_csv("/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/mappings.csv")
mapping = read_from_json("/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/all_locations.json")
mapping.update({'Negative': 18})
reverse_mapping = {str(v):k for k,v in mapping.items()}
df["LabelName"] = [reverse_mapping[str(l)] for l in df.Class]


### AP plot
# Plotting order = mean of each class
order = df.groupby(["LabelName"]).mean().AP.sort_values(ascending=True).index
df.LabelName = pd.Categorical(df.LabelName, 
                             ordered=True,
                             categories=order)
order = df.groupby(["LabelName"]).mean().AP.sort_values(ascending=True).index
g = ggplot(df, aes(x="LabelName", y="AP",fill="LabelName"))
g = g + geom_violin() + theme(axis_text_x = element_text(angle=90, hjust=1)) + geom_point()



### AP vs sample counts
gt_labels = pd.read_csv("/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/publictest/labels_publictest.csv")
gt_labels["Image_ID"] = [f.split("_")[0] for f in gt_labels.ID]
gt_labels["Cell_ID"] = [f.split("_")[1] for f in gt_labels.ID]
label_counts = get_location_counts(list(gt_labels.Label), mapping)
df_ = df.groupby(["LabelName"]).mean()
df_["LabelName"] = df_.index
df_["GT_count"] = [label_counts[l] for l in df_.index]
ggplot(df_, aes(x="LabelName", y ="GT_count", fill="LabelName")) + geom_bar(stat = 'identity')  + theme(axis_text_x = element_text(angle=90, hjust=1))
