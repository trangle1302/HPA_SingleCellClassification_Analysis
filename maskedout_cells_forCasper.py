#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:58:59 2021

@author: trangle
"""
import os
import io
import numpy as np
from imageio import imread, imwrite
import pandas as pd
import base64
import zlib
from pycocotools import _mask as coco_mask


def to_uint8(img):
    if img.dtype == np.dtype(np.uint16):
        img = np.clip(img, 0, 65535)
        img = (img / 65535 * 255.0).astype(np.uint8)
    elif img.dtype == np.dtype(np.float32) or img.dtype == np.dtype(np.float64):
        img = (img * 255).round().astype(np.uint8)
    elif img.dtype != np.dtype(np.uint8):
        raise Exception("Invalid image dtype " + img.dtype)
    return img

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def read_rgb(img_path, encoding=None):
    try:
        try:
            blue = imread(img_path + "_blue.png")
            yellow = imread(img_path + "_yellow.png")
            red = imread(img_path + "_red.png")
        except:
            blue = imread(os.path.join(img_path, "nuclei.png"))
            yellow = imread(os.path.join(img_path, "yellow.png"))
            red = imread(os.path.join(img_path, "microtubules.png"))
        red = np.clip(normalize(red) + normalize(yellow), 0, 1)
        green = normalize(yellow)
        img = np.dstack((to_uint8(red), to_uint8(green), to_uint8(normalize(blue))))

        if encoding:
            return encode_image_array(img, format=encoding)
        return img
    except:
        raise Exception(
            "At least one channel does not exist in " + os.path.basename(img_path)
        )
     

def encode_image_array(img, format="PNG"):
    in_mem_file = io.BytesIO()
    imwrite(in_mem_file, img, format=format)
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return base64.b64encode(img_bytes).decode("ascii")
   

def decodeToBinaryMask(rleCodedStr, imWidth, imHeight):
    # s = time.time()
    uncodedStr = base64.b64decode(rleCodedStr)
    uncompressedStr = zlib.decompress(uncodedStr, wbits=zlib.MAX_WBITS)
    detection = {"size": [imWidth, imHeight], "counts": uncompressedStr}
    detlist = []
    detlist.append(detection)
    mask = coco_mask.decode(detlist)
    binaryMask = mask.astype("uint8")[:, :, 0]
    # print(f'Decoding 1 cell: {time.time() - s} sec') #Avg 0.0035 sec for each cell
    return binaryMask 

def main():
    data_dir = '/data/kaggle-dataset/PUBLICHPA/images/test/'
    save_dir = '' # TODO: Update this!
    
    df = pd.read_csv('/data/kaggle-dataset/PUBLICHPA/mask/test.csv')
    
    imlist = list(df.ID.unique()) # TODO: Change to CAM images list
    for img in imlist:
        df_cells = df[df.ID == img]
        image = read_rgb(os.path.join(data_dir, img))
        for i,row in df_cells.iterrows():
            # Decode rle to binary mask
            mask = decodeToBinaryMask(row.cellmask, row.ImageWidth, row.ImageHeight)
            image_cell = image*mask # TODO: Either this works or you have to multiply each channel with mask
            
            imwrite(f'{save_dir}/{img}_{row.maskid}.png', image_cell)
        
if __name__ == '__main__':
    main()

 