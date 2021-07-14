import pandas as pd
from object_detection.metrics import oid_challenge_evaluation_utils as utils

if True:
    if True:  # FOrmatting the solution files, only need to be done once!
        all_annotations_hpa = pd.read_csv(
            "/home/trangle/HPA_SingleCellClassification/GT/_solution.csv_"
        )
        all_annotations = pd.DataFrame()
        for i, row in all_annotations_hpa.iterrows():
            pred_string = row.PredictionString.split(" ")
            for k in range(0, len(pred_string), 7):
                boxes = utils._get_bbox(
                    pred_string[k + 6], row.ImageWidth, row.ImageWidth
                )  # ymin, xmin, ymax, xmax
                line = {
                    "ImageID": row.ID,
                    "ImageWidth": row.ImageWidth,
                    "ImageHeight": row.ImageHeight,
                    "ConfidenceImageLabel": 1,
                    "LabelName": pred_string[k],
                    "XMin": boxes[1],
                    "YMin": boxes[0],
                    "XMax": boxes[3],
                    "YMax": boxes[2],
                    "IsGroupOf": pred_string[k + 5],
                    "Mask": pred_string[k + 6],
                }
                """
              binary_mask = decodeToBinaryMask(pred_string[k+6], row.ImageWidth, row.ImageHeight)
              plt.figure()
              plt.imshow(binary_mask)
              """
                all_annotations = all_annotations.append(line, ignore_index=True)
            print(f"{i}/{len(all_annotations_hpa)}, {len(all_annotations)} cells")
        all_annotations_hpa.to_csv(
            "/home/trangle/HPA_SingleCellClassification/all_annotations.csv",
            index=False,
        )
