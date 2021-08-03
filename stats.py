import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

d = "/home/trangle/HPA_SingleCellClassification/predictions"
df = pd.DataFrame()
teams = os.listdir(d)
teams = [t for t in teams if t!="OID" and os.path.isdir(os.path.join(d,t))]
for t in teams:
    files = os.listdir(os.path.join(d,t))
    output_f = [f for f in files if f.endswith("_outputmetrics.csv")]
    if len(output_f)==0:
        print(f"Team {t} has no mAP file!")
    else:
        df_ = pd.read_csv(os.path.join(d,t,output_f[0]), header=None)
        df_.columns = ["Metrics",t]
        df = pd.concat([df.reset_index(drop=True), df_], axis=1)
        print(df)

df.to_csv(os.path.join(d, "all_outputmetrics.csv"))