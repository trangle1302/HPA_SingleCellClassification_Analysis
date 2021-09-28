import os
import numpy as np
import glob
from imageio import imread, imwrite
from skimage.measure import regionprops
from collections import namedtuple
import pandas as pd
import base64
import zlib
from pycocotools import _mask as coco_mask
import time
import io

def find_coords(propslist, cell_label):
    for prop in propslist:
        if prop.label == cell_label:
            return prop.coords
    else:
        print(cell_label, "Cell Label Not found")
        return None


def iou_coords(X, Y):
    X = frozenset((tuple(x) for x in X))
    Y = frozenset((tuple(y) for y in Y))
    return len(X.intersection(Y)) / len(X.union(Y))


class TrackMax:
    def __init__(self, id=0, iou=0.0):
        self.id = id
        self.iou = 0.0


def cell_matching(predictions, realmasks):
    results = pd.DataFrame()
    for i, f in enumerate(predictions):
        image_name = os.path.basename(f).replace("_predictedmask.png", "")
        print(image_name)
        pred = imread(f)
        gt = imread(realmasks[i])
        # print(pred==gt)
        print(f"Number of cells predicted: {len(np.unique(pred))-1} {pred.max()}")
        print(f"Number of cells in groundtruth: {len(np.unique(gt))-1} {gt.max()}")

        cell_ids_p = np.unique(pred)[1:]
        cell_ids_g = np.unique(gt)[1:]
        overlap = set(cell_ids_g).intersection(set(cell_ids_p))

        ids_tomatch_g = set(cell_ids_g).difference(overlap)
        ids_tomatch_p = set(cell_ids_p).difference(overlap)

        regions_pred = regionprops(pred)
        regions_gt = regionprops(gt)

        maps_p = {}
        maps_g = {}
        for i in overlap:
            maps_p[i] = find_coords(regions_pred, i)
            maps_g[i] = find_coords(regions_gt, i)

        for i in list(overlap):
            g = maps_g[i]
            p = maps_p[i]
            iou = iou_coords(p, g)

            if iou == 0:
                ids_tomatch_g.add(i)
                ids_tomatch_p.add(i)
            else:
                result = {
                    "Image": image_name,
                    "GT cell label": i,
                    "Predicted cell label": i,
                    "IOU": iou,
                }
                results = results.append(result, ignore_index=True)
                print(result)

        print(ids_tomatch_g)
        if len(ids_tomatch_g) > 0:
            for i in ids_tomatch_g:
                g = find_coords(regions_gt, i)
                iou_max = TrackMax()
                for j in ids_tomatch_p:
                    p = find_coords(regions_pred, j)
                    p_iou = iou_coords(g, p)
                    if p_iou > iou_max.iou:
                        iou_max.iou = p_iou
                        iou_max.id = j

                result = {
                    "Image": image_name,
                    "GT cell label": i,
                    "Predicted cell label": iou_max.id,
                    "IOU": iou_max.iou,
                }
                print(result)
                results = results.append(result, ignore_index=True)

    return results


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


