import os
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from utils import *
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from tqdm import tqdm
from imageio import imread
from skimage.measure import regionprops
import time


def __main__(process_num=20):
    gt_mask_dir = "/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/data"
    gt_labels = pd.read_csv(
        "/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/labels.csv"
    )
    gt_labels["Image_ID"] = [f.split("_")[0] for f in gt_labels.ID]
    gt_labels["Cell_ID"] = [f.split("_")[1] for f in gt_labels.ID]

    # gt_mask_dir = "W:\home\trangle\Desktop\annotation-tool\HPA-Challenge-2020-all\data_for_Kaggle\data"
    # gt_labels = pd.read_csv("W:\Desktop\annotation-tool\HPA-Challenge-2020-all\data_for_Kaggle\labels.csv")
    save_dir = "/home/trangle/HPA_SingleCellClassification/examples/redai"
    # pred_mask_path = "/home/trangle/HPA_SingleCellClassification/examples/redai/redai_submission.csv"
    pred_mask_path = (
        "/home/trangle/HPA_SingleCellClassification/examples/redai/redai_submission.csv"
    )
    pred = pd.read_csv(pred_mask_path)
    s = time.time()
    os.makedirs(save_dir, exist_ok=True)
    print("Parent process %s." % os.getpid())
    p = Pool(process_num)
    for i in range(process_num):
        p.apply_async(
            run_proc,
            args=(
                pred,
                gt_mask_dir,
                gt_labels,
                save_dir,
                str(i),
                int(i * len(pred) / process_num),
                int((i + 1) * len(pred) / process_num),
            ),
        )
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print(f"All subprocesses done. {time.time()-s} sec")


def run_proc(pred_df, gt_mask_dir, gt_labels, save_dir, pid, sp, ep):
    print("Run child process %s (%s) sp:%d ep: %d" % (pid, os.getpid(), sp, ep))
    cell_matching(pred_df, gt_mask_dir, gt_labels, save_dir, pid, sp, ep)
    print("Run child process %s done" % (pid))


def cell_matching(pred_df, gt_mask_dir, gt_labels, save_dir, pid, sp, ep):
    results = pd.DataFrame()
    for i, row in tqdm(pred_df[sp:ep].iterrows(), postfix=pid):
        image_id = row.ID
        gt_masks = imread(os.path.join(gt_mask_dir, image_id + "_mask.png"))
        cell_idxes = set(np.unique(gt_masks)).difference([0])
        gt_labels_1image = gt_labels[gt_labels.Image_ID == image_id]

        assert set(gt_labels_1image.Cell_ID) == set([str(num) for num in cell_idxes])
        # Formating single cell's predictions
        width = row.ImageWidth
        height = row.ImageHeight
        pred_string = row.PredictionString.split(" ")
        preds = dict()
        for k in range(0, len(pred_string), 3):
            label = pred_string[k]
            conf = pred_string[k + 1]
            rle = pred_string[k + 2]
            if rle not in preds.keys():
                preds[rle] = dict()
            preds[rle].update({label: conf})

        """
        print(f'Number of cells predicted: {len(preds)}')
        print(f'Number of cells in groundtruth: {len(cell_idxes)} masks, {len(gt_labels_1image)} lab')
        for rle in preds.keys():
            binary_mask = decodeToBinaryMask(rle, width, height)
            plt.figure()
            plt.imshow(binary_mask[:,:,0])
        """
        gt_masks = regionprops(gt_masks)
        matched_ids = set()
        for rle in preds.keys():
            found = False
            binary_mask = decodeToBinaryMask(rle, width, height)
            p_coords = find_coords(regionprops(binary_mask), 1)

            # s = time.time()
            for cell_idx in cell_idxes - matched_ids:
                gt_coords = find_coords(gt_masks, cell_idx)
                iou = iou_coords(p_coords, gt_coords)
                if iou > 0.6:
                    matched_ids.add(cell_idx)
                    result = {
                        "Image": image_id,
                        "Cell_ID": cell_idx,
                        "GT cell label": gt_labels_1image[
                            gt_labels_1image.Cell_ID == str(cell_idx)
                        ].Label.values,
                        "Predicted cell label": preds[rle],
                        "IOU": iou,
                    }
                    # print(result)
                    results = results.append(result, ignore_index=True)
                    # print(f"Matching 1 cell takes {time.time() - s} sec")
                    found = True
                    continue

            if found:
                result = {
                    "Image": image_id,
                    "Cell_ID": None,
                    "GT cell label": None,
                    "Predicted cell label": preds[rle],
                    "IOU": 0,
                }
                results = results.append(result, ignore_index=True)

        cells_left = cell_idxes - matched_ids
        if len(cells_left) > 0:
            for cell_idx in cells_left:
                result = {
                    "Image": image_id,
                    "Cell_ID": cell_idx,
                    "GT cell label": gt_labels_1image[
                        gt_labels_1image.Cell_ID == str(cell_idx)
                    ].Label.values,
                    "Predicted cell label": None,
                    "IOU": 0,
                }
                results = results.append(result, ignore_index=True)

    results.to_csv(os.path.join(save_dir, f"{pid}.csv"))


if __name__ == "__main__":
    __main__()
