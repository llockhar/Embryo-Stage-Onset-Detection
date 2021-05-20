# -*- coding: utf-8 -*-
import numpy as np
import os
from os.path import join, exists
from glob import glob
import matplotlib.pyplot as plt
import argparse

from skimage.io import imread, imsave
from skimage.feature import canny
from skimage.measure import regionprops
from skimage.morphology import square, binary_dilation, binary_opening

parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, help="Path to unprocessed images")
parser.add_argument("--out_path", type=str, help="Desired output path for pre-processed images")
args = parser.parse_args()

IMG_PATH = args.img_path
OUT_PATH = args.out_path

if not exists(OUT_PATH):
    os.makedirs(OUT_PATH)

def get_centroid_from_mask(in_img, in_mask):
    edges = canny(in_img, sigma=1.0, low_threshold=(255*0.15), high_threshold=(255*0.05), mask=(in_mask>0))
    edges = binary_dilation(edges, selem=square(3))
    edges = binary_opening(edges, selem=square(4))

    props = regionprops(edges.astype(np.uint8))
    centroid_coords = props[0].centroid

    return centroid_coords

def get_crop_and_mask(in_img, centroid, rad):
    M,N = in_img.shape
    height = int(centroid[0])
    width  = int(centroid[1])

    pad_h_before = 0
    pad_h_after = 0
    pad_w_before = 0
    pad_w_after = 0

    if height < rad:
        pad_h_before = rad - height
        height += pad_h_before
    elif height > (M - rad):
        pad_h_after = height - M + rad

    if width < rad:
        pad_w_before = rad - width
        width += pad_w_before
    elif width > (N - rad):
        pad_w_after = width - N + rad

    in_img = np.pad(in_img, ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)), mode='constant')

    circ_mask = np.zeros((rad*2,rad*2)).astype(np.bool)
    xx, yy = np.meshgrid(np.linspace(0,rad*2,rad*2), np.linspace(0,rad*2,rad*2))
    circ_mask = circ_mask | (np.sqrt(abs(xx - rad)**2 + abs(yy - rad)**2) <= rad)
    circ_mask = (circ_mask*1).astype(np.uint8)

    crop_img = in_img[height-rad:height+rad, width-rad:width+rad]

    return crop_img, circ_mask


for img_name in glob(IMG_PATH + '*/*.JPG'):
    if 'R' in img_name:
        img = imread(img_name)
        print(img_name)

        M,N = img.shape
        circ_mask1 = np.zeros((M,N)).astype(np.bool)
        xx, yy = np.meshgrid(np.linspace(0,N,N), np.linspace(0,M,M))
        circ_mask1 = circ_mask1 | (np.sqrt(abs(xx - N/2)**2 + abs(yy - M/2)**2) <= 200)

        centroid1 = get_centroid_from_mask(img, circ_mask1)
        img, circ_mask2 = get_crop_and_mask(img, centroid1, rad=200)
        centroid2 = get_centroid_from_mask(img, circ_mask2)
        crop_img, circ_mask2 = get_crop_and_mask(img, centroid2, rad=190)

        out_image = (crop_img * circ_mask2).astype(np.uint8)
        
        # plt.imshow(out_image)
        # plt.show(block=False)
        # plt.pause(1)

        if not exists(OUT_PATH + img_name.split('\\')[-2]):
            os.mkdir(OUT_PATH + img_name.split('\\')[-2])
        imsave(join(OUT_PATH, img_name.split('\\')[-2], img_name.split('\\')[-1]), out_image)
