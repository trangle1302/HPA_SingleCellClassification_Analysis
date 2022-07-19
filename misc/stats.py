from io import DEFAULT_BUFFER_SIZE
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _load_labelmap(labelmap_path):
    """Loads labelmap from the labelmap path.

  Args:
    labelmap_path: Path to the labelmap.

  Returns:
    A dictionary mapping class name to class numerical id
    A list with dictionaries, one dictionary per category.
  """

    label_map = string_int_label_map_pb2.StringIntLabelMap()
    with open(labelmap_path, "r") as fid:
        label_map_string = fid.read()
        text_format.Merge(label_map_string, label_map)
    labelmap_dict = {}
    categories = []
    for item in label_map.item:
        labelmap_dict[item.name] = item.id
        categories.append({"id": item.id, "name": item.name})
    return labelmap_dict, categories


d = "/data/kaggle-dataset/mAPscoring"
plot_dir = "/home/trangle/HPA_SingleCellClassification/plots"
mapping = (
    "/home/trangle/HPA_SingleCellClassification/GT/hpa_single_cell_label_map.pbtxt"
)
class_label_map, categories = _load_labelmap(FLAGS.input_class_labelmap)

if True:
    df = None
    teams = os.listdir(d)
    teams = [t for t in teams if t != "OID" and os.path.isdir(os.path.join(d, t))]

    for t in teams:
        files = os.listdir(os.path.join(d, t))
        output_f = [f for f in files if f.endswith("_outputmetrics.csv")]
        if len(output_f) == 0:
            print(f"Team {t} has no mAP file!")
        else:
            df_ = pd.read_csv(os.path.join(d, t, output_f[0]), header=None)
            df_.columns = ["Metrics", t]
            if df is None:
                df = df_
            else:
                df = df.merge(df_, on="Metrics")

    df.index = df.Metrics
    df = df.drop("Metrics", axis=1)
    df.to_csv(os.path.join(d, "all_outputmetrics.csv"))

df = pd.read_csv(os.path.join(d, "all_outputmetrics.csv"))
print(df.columns)
dfT = df.T
# print(dfT.head())
plt.figure()
plt.plot("OpenImagesDetectionChallenge_Precision/mAP@0.6IOU", data=dfT)
# sns.barplot(x="OpenImagesDetectionChallenge_Precision/mAP@0.6IOU", y="Metrics", data=dfT)
plt.savefig(os.path.join(plot_dir, "mAP"))
