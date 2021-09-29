import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import regionprops#, regionprops_table
import imageio
import seaborn as sns
import utils

BASE_DIR = "/home/trangle/HPA_SingleCellClassification/hpa2021_0902"
IMAGE_DIR = "/data/kaggle-dataset/PUBLICHPA"
SAVE_DIR = "/home/trangle/HPA_SingleCellClassification/plots"

LABEL_TO_NAME = dict({
  0: 'Nucleoplasm',
  1: 'Nuclear membrane',
  2: 'Nucleoli',
  3: 'Nucleoli fibrillar center',
  4: 'Nuclear speckles',
  5: 'Nuclear bodies',
  6: 'Endoplasmic reticulum',
  7: 'Golgi apparatus',
  8: 'Intermediate filaments',
  9: 'Actin filaments',
  10: 'Microtubules',
  11: 'Mitotic spindle',
  12: 'Centrosome',
  13: 'Plasma membrane',
  14: 'Mitochondria',
  15: 'Aggresome',
  16: 'Cytosol',
  17: 'Vesicles and punctate cytosolic patterns',
  18: 'Negative',
})
LABEL_TO_ALIAS = dict({
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
})
ALIAS_TO_LABEL = {str(v):k for k,v in LABEL_TO_ALIAS.items()}

df = pd.read_csv(os.path.join(BASE_DIR,"result/d0507_cellv4b_3labels_cls_inception_v3_cbam_i128x128_aug2_5folds/fold0/epoch_12.00_ema/cell_result_train_cell_v1.csv"))
csv = pd.read_csv(os.path.join(IMAGE_DIR,"mask/train.csv"))
#df = pd.merge(df,csv[['ImageHeight', 'ImageWidth']], on="'ImageHeight', 'ImageWidth'")
metadata = pd.read_csv("/data/HPA-IF-images/IF-image.csv")
metadata["Image_ID"] = [os.path.basename(f)[:-1] for f in metadata.filename]
"""
for (k,v) in ALIAS_TO_LABEL.items():
    plt.figure()n
    plt.hist(df[k], bins=1000, alpha = 0.3)
    plt.xlabel("Class confidence score")
    plt.ylabel("Cell counts")
    plt.title(LABEL_TO_NAME[v])
    plt.savefig(os.path.join(SAVE_DIR,f"confidence_by_bestfittinghpasc_for_PUBLICHPA_{k}.png"))
    plt.close()
"""
    
THRESHOLD = 0.3

imlist = list(set(df.ID))
for im in imlist[:10]:  
  pcdf = df[df.ID==im]
  pcdf.to_csv(os.path.join(SAVE_DIR,f"predicted_cells_{im}.csv"), index=False)
  cdf = csv[csv.ID==im]
  cdf.to_csv(os.path.join(SAVE_DIR,f"train_cells_{im}.csv"), index=False)
  img = utils.read_rgb(os.path.join(IMAGE_DIR,"images","train",im))
  cellmask = imageio.imread(os.path.join(IMAGE_DIR,"mask","train",im+"_cellmask.png"))
  nucleimask = imageio.imread(os.path.join(IMAGE_DIR,"mask","train",im+"_nucleimask.png"))
  props = regionprops(cellmask)
  fig, ax = plt.subplots(1,3, figsize=(30,10))
  ax[0].imshow(img)
  ax[0].set_title("RGB",fontsize=18)
  ax[0].axis('off')
  ax[1].imshow(plt.imread(os.path.join(IMAGE_DIR,"images","train",im+'_green.png')))
  ax[1].set_title("Protein channel",fontsize=18)
  ax[1].axis('off')
  ax[2].imshow(cellmask, alpha=1)
  ax[2].imshow(nucleimask, alpha=0.5)
  ax[2].set_title("Mask & Prediction",fontsize=18)
  ax[2].axis('off')
  for i, row in pcdf.iterrows():
      bbox = [p.bbox for p in props if p.label==row.maskid]
      labels = [k for k in ALIAS_TO_LABEL.keys() if row[k]>THRESHOLD]
      if ('Negative' in labels) or len(labels)==0:
          labels = 'Negative'
      else:
          labels= '\n'.join(labels)   
      x = (bbox[0][1] + bbox[0][3])/2
      y = (bbox[0][0] + bbox[0][2])/2
      ax[2].text(x,y+100, labels, color='white')
  image_locations = metadata[metadata.Image_ID==im].locations.values[0]
  fig.suptitle(f"{im} : {image_locations}", fontsize=16)
  plt.savefig(os.path.join(SAVE_DIR,"psc",im+'.jpg'))



def plot_():
    fig, ax = plt.subplots(1,2, figsize=(20,10)) 
    ax[0].imshow(img)
    ax[0].axis('off')
    for i, row in cdf.iterrows():
        cell_id = int(row.cell_id.split('/')[1])
        poly = mapping[cell_id]
        ax[0].add_patch(PolygonPatch(poly, fc='#6699cc', ec='#6699cc', alpha=0.5, zorder=2 ))    
        x = (poly['bbox'][0][1] + poly['bbox'][0][3])/2
        y = (poly['bbox'][0][0] + poly['bbox'][0][2])/2
        ax[0].text(x,y, row.labels, color='white')
    ax[1].imshow(imread(img_path + '_green.png'))
    ax[1].axis('off')
    
    plt.title('Protein channel')
    plt.suptitle(f'{image_id}, Image annotation: {row.image_locations} \n {subtext}')   
    #plt.tight_layout()
    #plt.show()  
    plt.savefig(os.path.join(DATA_DIR,'final_label_check', image_id + '.png'))
    print(f'{image_id} done plotting')