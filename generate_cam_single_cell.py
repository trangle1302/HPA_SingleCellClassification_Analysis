import glob
import imageio
from skimage.measure import regionprops

MASK_DIR="/data/kaggle-dataset/CAM_images/mask/train"
IMAGE_DIR="/data/kaggle-dataset/CAM_images/2nd_cams"
SAVE_DIR="/data/kaggle-dataset/CAM_images/2nd_cams"

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
    imlist = glob.glob(f"{IMAGE_DIR}/")
    masklist = glob.glob(f"{MASK_DIR}/*_cellmask.png")
    print(f"Found {len(masklist)} masks")
    for mask_path in masklist:
        mask = imageio.imread(mask_path)
        img = imageio.imread(glob.glob(f"{IMAGE_DIR}/{img_name}_*.png"))
        regions = regionprops(mask)
        for region in regions:
            bbox = region.bbox
            cell_id = region.label
            cell = img[bbox[0]:bbox[1],bbox[2]:bbox[3],:]
            imageio.imwrite(f"{SAVE_DIR}/{img_name}_{cell_id}_{location}.png", cell)

if __name__ == '__main__':
    main()
