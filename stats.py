from io import DEFAULT_BUFFER_SIZE
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

d = "/home/trangle/HPA_SingleCellClassification/predictions"
plot_dir = "/home/trangle/HPA_SingleCellClassification/plots"
mapping = "/home/trangle/HPA_SingleCellClassification/GT/hpa_single_cell_label_map.pbtxt"

if False:
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
    df = df.drop('Metrics', axis = 1)
    df.to_csv(os.path.join(d, "all_outputmetrics.csv"))

df = pd.read_csv(os.path.join(d, "all_outputmetrics.csv"))
dfT = df.T
print(dfT.head())
plt.figure()
plt.plot("OpenImagesDetectionChallenge_Precision/mAP@0.6IOU", data=dfT)
#sns.barplot(x="OpenImagesDetectionChallenge_Precision/mAP@0.6IOU", y="Metrics", data=dfT)
plt.savefig(os.path.join(plot_dir,"mAP"))