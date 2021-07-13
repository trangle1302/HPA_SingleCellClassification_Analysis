import os
import pandas as pd


d = "/home/trangle/HPA_SingleCellClassification/examples/bestfitting"
files = [f for f in os.listdir(d) if not f.endswith("_submission.csv")]
df = pd.DataFrame()
for f in files:
    df_ = pd.read_csv(os.path.join(d, f))
    df = df.append(df_, ignore_index=True)
