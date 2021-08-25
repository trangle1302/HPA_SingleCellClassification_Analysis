import os
import pandas as pd

folders = {
    "17th": {
        "public": "/data/kaggle-dataset/mAPscoring/17th/HPA_Fast_Sub_20981336.csv",
        "private": "/data/kaggle-dataset/mAPscoring/17th/HPA_Inference_Final_Sub_1.csv",
    },
    "19th": {
        "public": "/data/kaggle-dataset/mAPscoring/19th/quicksub_20982022.csv",
        "private": "/data/kaggle-dataset/mAPscoring/19th/pri_4models.csv",
    },
    "22nd": {
        "public": "/data/kaggle-dataset/mAPscoring/22nd/HPA_CT_ILL_Inference_Public.csv",
        "private": "/data/kaggle-dataset/mAPscoring/22nd/HPA_CT_ILL_Inference_Private.csv",
    },
    "24th": {
        "public": "/data/kaggle-dataset/mAPscoring/24th/1res50_1xcep_1024_1090_nopp_75-100-125-150_20917918.csv",
        "private": "/data/kaggle-dataset/mAPscoring/24th/1res1xcep_1024_2080_nopp_75-100_private.csv",
    },
}

pub_imgs = pd.read_csv(
    "/home/trangle/HPA_SingleCellClassification/GT/all_annotations_publictest.csv"
)
pub_imgs = list(set(pub_imgs.ImageID))

pri_imgs = pd.read_csv(
    "/home/trangle/HPA_SingleCellClassification/GT/all_annotations_privatetest.csv"
)
pri_imgs = list(set(pri_imgs.ImageID))

for f in folders.keys():
    pub = pd.read_csv(folders[f]["public"])
    pub = pub[pub.ID.isin(pub_imgs)]
    pri = pd.read_csv(folders[f]["private"])
    pri = pri[pri.ID.isin(pri_imgs)]
    merged = pd.concat([pub, pri])
    print(merged.columns, merged.shape)
    save_path = os.path.join(
        "/data/kaggle-dataset/mAPscoring", f, "all_test_submission.csv"
    )
    print(f"Done merging public and private submissions, saving to {save_path}")
    merged.to_csv(save_path, index=False)
