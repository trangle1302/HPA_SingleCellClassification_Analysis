#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:54:23 2021

@author: trangle
This code is to prepare HPA public images for single cell prediction by bestfitting model
"""

import os
import pandas as pd
import shutil
import json
import io
from requests.auth import HTTPBasicAuth
import requests
import imageio
import gzip

base_url = 'https://if.proteinatlas.org'

def move_publichpa(ifimage, data_dir, save_dir, channels):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    log = open("/data/kaggle-dataset/PUBLICHPA/log.txt", "a+")
    for i, row in ifimage.iterrows():
        filename = row.filename
        new_name = filename.replace("/archive", data_dir)
        for ch in channels:
            img_path = new_name + ch
            dest_path = os.path.join(save_dir, filename.split("/")[-1] + ch)
            if os.path.exists(dest_path):
                continue

            try:
                shutil.copy(img_path, dest_path)
                print(f"Copied {img_path}.......{dest_path}")
            except:
                if os.path.exists(img_path):
                    print(f"{img_path} does not exist")
                    f.write(img_path)
                else:
                    try:       
                        url = filename.replace("/archive", base_url) + ch
                        url = url.replace(".png", ".tif.gz")
                        r = requests.get(url, auth=HTTPBasicAuth('trang', 'H3dgeh0g'))
                        print(url)
                        if r.status_code != 200:
                            url = url.replace(".png", ".tif")
                            r = requests.get(url, auth=HTTPBasicAuth('trang', 'H3dgeh0g'))
                            tf = r.content
                            #print(r.status_code, url)
                        else:
                            f = io.BytesIO(r.content)
                            tf = gzip.open(f).read()
                        img = imageio.imread(tf, 'tiff')
                        imageio.imwrite(dest_path, img)
                    except:
                        log.write(filename)
                        pass
    log.close()
            

def move_imglist(imlist, data_dir, save_dir, channels):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for img in imlist:
        (plate, well, sample) = img.split(":")
        new_name = img.replace(":", "_")
        for ch in channels:
            dest_path = os.path.join(save_dir, new_name + "_" + ch)
            if not os.path.exists(dest_path):
                img_path = os.path.join(data_dir, plate, new_name + "_" + ch)
                shutil.copy(img_path, dest_path)

def read_from_json(json_file_path):
    """Function to read json file (annotation file)
    """
    with io.open(json_file_path, "r", encoding="utf-8-sig") as myfile:
        data = json.load(myfile)
    return data


def add_label_idx(df, all_locations):
    '''Function to convert label name to index
    '''
    df["Label"] = None
    for i, row in df.iterrows():
        locations = str(row.locations)
        if locations == 'nan':
            df.loc[i,"Label"] = "18"
            continue
        labels = locations.split(',')
        idx = []
        for l in labels:
            if l in all_locations.keys():
                idx.append(str(all_locations[l]))
        if len(idx)>0:
            df.loc[i,"Label"] = "|".join(idx)
    return df


def main():
    CHANNELS = ["red.png", "green.png", "blue.png", "yellow.png"]
    DATA_DIR = "/data/HPA-IF-images"
    SAVE_DIR = "/data/kaggle-dataset/PUBLICHPA/images/test"
    CAM_IMG_DIR = "/data/kaggle-dataset/CAM_images"
    
    all_locations = read_from_json("/home/trangle/HPA_SingleCellClassification/all_locations_merged.json")
    
    ifimages = pd.read_csv("/data/HPA-IF-images/IF-image.csv")
    #ifimages_v21 = ifimages[ifimages.latest_version == 21.0]
    ifimages['v21'] = ['21.0' in str(r.versions).split(',') for _,r in ifimages.iterrows()]
    ifimages_v21 = ifimages[ifimages.v21==True]

    genelist = [f.split(',') for f in ifimages_v21.ensembl_ids]
    genelist = [item for sublist in genelist for item in sublist]
    print(f'Number of images {ifimages_v21.shape[0]}, {len(set(genelist))} genes')

    print(ifimages_v21.shape)
    ifimages_v21 = add_label_idx(ifimages_v21, all_locations)
    print(ifimages_v21.shape)

    ifimages_v21["ID"] = [os.path.basename(f)[:-1] for f in ifimages_v21.filename]
    ifimages_v21_ = ifimages_v21[["ID","Label","gene_names","ensembl_ids","atlas_name","locations"]]
    #ifimages_v21_ = ifimages_v21[ifimages_v21.Label.isna() == False]
    print(ifimages_v21_[ifimages_v21.Label.isna()])
    ifimages_v21_.to_csv(os.path.join(SAVE_DIR.replace("images", "raw"), "test.csv"), index=False)
        
    
    move_publichpa(ifimages_v21, DATA_DIR, SAVE_DIR, CHANNELS)
    
    # Hand pick images list for CAM
    imlist = ["1878:C8:33", "1657:E10:4", "1717:G8:34", "333:B6:1", "836:E7:2", "1326:C6:5", "465:F10:1","152:B7:1", "535:C3:1", "1038:G7:1", "570:F10:1","476:C7:1", "1537:F3:2","492:D3:2", "814:C5:1"]
    #move_imglist(imglist, DATA_DIR, CAM_IMG_DIR)
    
if __name__ == '__main__':
    main()
