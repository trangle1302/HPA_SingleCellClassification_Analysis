import os
import pandas as pd
import numpy as np
import argparse
import tqdm
import sys
import traceback


def filter_pred_str(pred_string, c):
    pred_string = pred_string.split(" ")
    items = []
    for k in range(0, len(pred_string), 3):
        label = pred_string[k]
        # print(c, label, str(label) == str(c))
        if str(label) == str(c):
            items += [pred_string[k], pred_string[k + 1], pred_string[k + 2]]
            # print(items)
    pred_string_filtered = " ".join(items)
    return pred_string_filtered


parser = argparse.ArgumentParser(description="Creating bash script")
parser.add_argument("-file", type=str, help="path to submission file")
args = parser.parse_args()


d = os.path.dirname(args.file)
submissions = pd.read_csv(args.file)
classes = np.arange(0, 19)

for c in classes:
    if os.path.exists(args.file.replace(".csv", f"_{c}.csv")):
        print("removing existing class submission file....")
        os.remove(args.file.replace(".csv", f"_{c}.csv"))
    f = open(args.file.replace(".csv", f"_{c}.csv"), "a+")
    f.write("ID,ImageWidth,ImageHeight,PredictionString\n")
    print(f"{submissions.shape[0]} images predicted, formatting for class {c}...")
    for i, row in tqdm.tqdm(submissions.iterrows(), total=submissions.shape[0]):
        try:
            pred_str = filter_pred_str(row.PredictionString, c)
            line = f"{row.ID},{row.ImageWidth},{row.ImageHeight},{pred_str}\n"
            f.write(line)

        except Exception as e:
            e = sys.exc_info()
            print("Error Return Type: ", type(e))
            print("Error Class: ", e[0])
            print("Error Message: ", e[1])
            print("Error Traceback: ", traceback.format_tb(e[2]))
            continue
    f.close()
