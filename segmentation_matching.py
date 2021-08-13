import os
import pandas as pd
import numpy as np
from scipy.ndimage.measurements import mean
from sklearn.metrics import jaccard_similarity_score
from utils import *
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from tqdm import tqdm
from imageio import imread
from skimage.measure import regionprops
import time
import argparse

parser = argparse.ArgumentParser(description="Creating bash script")
parser.add_argument("-file", type=str, help="path to submission file")
args = parser.parse_args()


def __main__(args, process_num=10):
    gt_mask_dir = "/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/data"
    gt_labels = pd.read_csv(
        "/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/labels.csv"
    )
    gt_labels["Image_ID"] = [f.split("_")[0] for f in gt_labels.ID]
    gt_labels["Cell_ID"] = [f.split("_")[1] for f in gt_labels.ID]
    print(f"GT labels have {len(gt_labels)} cells")
    # gt_mask_dir = "W:\home\trangle\Desktop\annotation-tool\HPA-Challenge-2020-all\data_for_Kaggle\data"
    # gt_labels = pd.read_csv("W:\Desktop\annotation-tool\HPA-Challenge-2020-all\data_for_Kaggle\labels.csv")
    save_dir = os.path.dirname(args.file)
    # pred_mask_path = "/home/trangle/HPA_SingleCellClassification/predictions/redai/redai_submission.csv"
    pred_mask_path = args.file
    pred = pd.read_csv(pred_mask_path)
    pred = pred[pred.ID.isin(gt_labels.Image_ID)]
    meta = pd.read_csv(
        "/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/images_metadata.csv"
    )

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
    print(f"Finished in {(time.time() - s)/3600} h")
    merge_df(save_dir, meta)


def run_proc(pred_df, gt_mask_dir, gt_labels, save_dir, pid, sp, ep):
    print("Run child process %s (%s) sp:%d ep: %d" % (pid, os.getpid(), sp, ep))
    cell_matching(pred_df, gt_mask_dir, gt_labels, save_dir, pid, sp, ep)
    print("Run child process %s done" % (pid))


def cell_matching(pred_df, gt_mask_dir, gt_labels, save_dir, pid, sp, ep):
    results = pd.DataFrame()
    f = open(os.path.join(save_dir, f"IOU_part_{pid}.csv"), "a+")  
    f.write("Image,Cell_ID,GT_cell_label,Predicted_cell_label,IOU\n")
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
                    gt_lab = gt_labels_1image[gt_labels_1image.Cell_ID == str(cell_idx)].Label.values
                    line = f"{image_id},{cell_idx},{gt_lab},{preds[rle]},{iou}\n"
                    """
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
                    """
                    f.write(line)
                    found = True
                    continue

            if found:
                line = f"{image_id},{None},{None},{preds[rle]},{0}\n"
                """
                result = {
                    "Image": image_id,
                    "Cell_ID": None,
                    "GT cell label": None,
                    "Predicted cell label": preds[rle],
                    "IOU": 0,
                }
                results = results.append(result, ignore_index=True)
                """
                f.write(line)

        cells_left = cell_idxes - matched_ids
        if len(cells_left) > 0:
            for cell_idx in cells_left:
                gt_lab = gt_labels_1image[gt_labels_1image.Cell_ID == str(cell_idx)].Label.values
                line = f"{image_id},{cell_idx},{gt_lab},{None},{0}\n"
                """
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
                """
                f.write(line)

    #results.to_csv(os.path.join(save_dir, f"IOU_part_{pid}.csv"))

    f.close()

def merge_df(d, meta):
    files = [f for f in os.listdir(d) if f.startswith("IOU_part_")]
    print(files)
    df = pd.DataFrame()
    for f in files:
        df_ = pd.read_csv(os.path.join(d, f))
        df = df.append(df_, ignore_index=True)
    df_merged = df.merge(meta, right_on="Image_ID", left_on="Image")
    df_merged.to_csv(os.path.join(d, f"IOU_merged.csv"))
    df_ = df_merged[df_merged["GT cell label"] != None]
    # print(df_.columns)
    # print(f"matched {sum(df_.Cell_ID.isna()==False)}/{len(df_)} cells")
    df_ = df_.groupby(["Image", "Cell_ID"])
    print(f"Mean IOU: {df_.IOU.mean().values}, {len(df_)} cells")


if __name__ == "__main__":
    __main__(args)
