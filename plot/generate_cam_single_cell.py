import glob
import os
import cv2
import imageio
from skimage.measure import regionprops

MASK_DIR="/data/kaggle-dataset/CAM_images/mask/train"
IMAGE_DIR="/data/kaggle-dataset/CAM_images/3rd_cams/cams"
SAVE_DIR="/data/kaggle-dataset/CAM_images/3rd_cams"

LABEL_TO_ALIAS = {
    0: 'Nucleoplasm',
    1: 'NuclearM',
    2: 'Nucleoli',
    3: 'NucleoliFC',
    4: 'NuclearS',
    5: 'NuclearB',
    6: 'EndoplasmicR',
    7: 'GolgiA',
    8: 'IntermediateF',
    9: 'ActinF',
    10: 'Microtubules',
    11: 'MitoticS',
    12: 'Centrosome',
    13: 'PlasmaM',
    14: 'Mitochondria',
    15: 'Aggresome',
    16: 'Cytosol',
    17: 'VesiclesPCP',
    18: 'Negative',
}

def main():
    masklist = glob.glob(f"{MASK_DIR}/*_cellmask.png")
    print(f"Found {len(masklist)} masks, example path {masklist[0]}")
    for mask_path in masklist:
        img_name = os.path.basename(mask_path).replace("_cellmask.png","")
        mask = imageio.imread(mask_path)
        img_paths = glob.glob(f"{IMAGE_DIR}/{img_name}_*.png")
        for img_path in img_paths:
            #print(f"{IMAGE_DIR}/{img_name}_*.png",img_path)
            location = img_path.split("_")[-1].replace(".png", "")
            img = imageio.imread(img_path)
            img = cv2.resize(img, (2048,2048))
            #print(f"Mask shape: {mask.shape}, Image shape: {img.shape}")
            regions = regionprops(mask)
            for region in regions:
                bbox_ = region.bbox
                cell_id = region.label
                cell = img[bbox_[0]:bbox_[2],bbox_[1]:bbox_[3],:]
                """
                #cell = cv2.resize(cell, (128,128))
                plt.figure()
                plt.imshow(cell)
                plt.imshow(mask[bbox_[0]:bbox_[2],bbox_[1]:bbox_[3]]==cell_id, alpha=0.3)
                break
                """
                imageio.imwrite(f"{SAVE_DIR}/{img_name}_{cell_id}_{location}.png", cell)

if __name__ == '__main__':
    main()
