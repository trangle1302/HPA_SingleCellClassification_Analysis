import os
import pandas as pd
from sklearn.metrics import jaccard_similarity_score
from utils import *
import matplotlib.pyplot as plt

def __main__():
    gt_mask_dir = "/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/data"
    gt_labels = pd.read_csv("/home/trangle/Desktop/annotation-tool/HPA-Challenge-2020-all/data_for_Kaggle/labels.csv")
    #gt_mask_dir = "W:\home\trangle\Desktop\annotation-tool\HPA-Challenge-2020-all\data_for_Kaggle\data"
    #gt_labels = pd.read_csv("W:\Desktop\annotation-tool\HPA-Challenge-2020-all\data_for_Kaggle\labels.csv")
    
    pred_mask_path = "/home/trangle/HPA_SingleCellClassification/examples/bestfitting_submission.csv"
    
    pred = pd.read_csv(pred_mask_path)
    #matches = cell_matching()

    for i, row in pred.iterrows():
        image_id = row.ID
        gt_masks = plt.imread(os.path.join(gt_mask_dir, image_id+'_mask.png'))
        
        # Formating the se
        width = row.ImageWidth
        height = row.ImageHeight
        pred_string = row.PredictionString.split(" ")
        preds = dict()
        for k in range(0, len(pred_string), 3): 
            label = pred_string[k]
            conf = pred_string[k+1]
            rle = pred_string[k+2]
            if rle not in preds.keys():
                preds[rle] = dict()
            preds[rle].update({label: conf})
        
        print(f'Number of cells predicted: {len(np.unique(pred))-1} {pred.max()}')
        print(f'Number of cells in groundtruth: {len(np.unique(gt))-1} {gt.max()}')
    
        for rle in preds.keys():
            binary_mask = decodeToBinaryMask(rle, width, height)
            plt.figure()
            plt.imshow(binary_mask[:,:,0])
            
        
        
if __name__ == "__main__":
    __main__()