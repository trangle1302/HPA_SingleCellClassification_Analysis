import os
import pandas as pd
import numpy as np


def average_precisions(y_true, y_pred):
    _, classes = y_true.shape
    average_precisions = []

    for index in range(classes):
        row_indices_sorted = np.argsort(-y_pred[:, index])

        y_true_cls = y_true[row_indices_sorted, index]
        y_pred_cls = y_pred[row_indices_sorted, index]

        tp = y_true_cls == 1
        fp = y_true_cls == 0

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        npos = np.sum(y_true_cls)

        rec = tp * 1.0 / npos

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp * 1.0 / np.maximum((tp + fp), np.finfo(np.float64).eps)

        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            average_precisions.append(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

    return average_precisions

def main(d):
    files = [f for f in os.listdir(d) if not f.endswith("_submission.csv")]
    df = pd.DataFrame()
    for f in files:
        df_ = pd.read_csv(os.path.join(d, f))
        df = df.append(df_, ignore_index=True)

    # df['GT cell label'] = [f.split("\n")[0].split(" ")[-1] for f in df['GT cell label']]

if __name__ == "__main__":
    d = "/home/trangle/HPA_SingleCellClassification/examples/bestfitting"
    main(d)