import pandas as pd
import imageio
import glob
import cv2


df = pd.read_csv('test_df_cams/result_df_singlelabel.csv')
df['id'] = ["_".join([r.image_id,str(r.cell_id),r.LabelName]) for _,r in df.iterrows()]
tmp = df[df.iou_all==1]

img = imageio.imread(glob.glob(f"test_df_cams/c7360b95-bb35-452c-ad2c-2693aba7c38c_5_Negativ*.png")[0])
green = img[:128,:128]
imageio.imsave("l_green.png", green)
cam = img[:128, 128:]
imageio.imsave("l_cam.png", cam)
bgreen = img[128:,:128]
imageio.imsave("l_bgreen.png", bgreen)
_,bcam = cv2.threshold(cam, 30, 255, cv2.THRESH_BINARY)
imageio.imsave("l_bcam.png", bcam)