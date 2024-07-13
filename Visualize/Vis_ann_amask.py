import os
import numpy as np
import cvbase as cvb
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import pdb
import copy
import skimage.io as io
import matplotlib.pyplot as plt
import random

def polys_to_mask(polygons, height, width):
	rles = mask_utils.frPyObjects(polygons, height, width)
	rle = mask_utils.merge(rles)
	mask = mask_utils.decode(rle)
	return mask

def vis_mask(img_name, img, a_m, obj_id, vis=False, save_path=None):
	# cv2.namedWindow("a_m")
	a_i = copy.deepcopy(img)
	a_m = copy.deepcopy(a_m)

	a_m = a_m.astype(np.uint8) * 255
	a_m = np.stack((a_m, a_m, a_m), axis=2)
	a_m_w = cv2.addWeighted(a_i, 0.4, a_m, 0.6, 0)

	out = np.concatenate((img, a_m_w), axis=0)
	cv2.putText(out, 'Current Image is: ' + img_name, (100, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))
	cv2.putText(out, 'Current Object is:   ' + str(obj_id), (100, 430), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255))
	h,w,_ = out.shape
	out = cv2.resize(out, dsize=(int(w*1.2), int(h*1.2)),interpolation=cv2.INTER_NEAREST)
	if vis:
		cv2.imshow("amask", out)
		cv2.waitKey()
	if save_path is not None:
		save_name = os.path.join(save_path, img_name[:-4]+'obj_'+str(obj_id)+'.png')
		cv2.imwrite(save_name, out)
		print('Save object {} of img {}'.format(obj_id, img_name))


def make_json_dict(imgs, anns):
	imgs_dict = {}
	anns_dict = {}
	for ann in anns:
		image_id = ann["image_id"]
		if not image_id in anns_dict:
			anns_dict[image_id] = []
			anns_dict[image_id].append(ann)
		else:
			anns_dict[image_id].append(ann)
	
	for img in imgs:
		image_id = img['id']
		imgs_dict[image_id] = img['file_name']

	return imgs_dict, anns_dict

is_train = False


base_img_path = "../BlendPASS/leftImg8bit/val"
base_ann_path = "../BlendPASS/annotations/val.json"
anns = cvb.load(base_ann_path)
imgs_info = anns['images']
anns_info = anns["annotations"]

imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)

color_list = [255]
Checked_id = 0
Vis_results = True
Save_vis_img_path = None
for img_id in anns_dict.keys():
	img_name = imgs_dict[img_id]
	print(img_name)
	img_path = os.path.join(base_img_path, img_name)
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)

	height, width, _ = img.shape
	anns = anns_dict[img_id]
	a_masks = {}
	for ann in anns:
		a_mask = polys_to_mask(ann["segmentation"], height, width)
		obj_id = ann["id"]
		if Checked_id >= obj_id:
			continue
		a_masks[obj_id]=a_mask
	for k, v in a_masks.items():
		vis_mask(img_name, img, v, k, Vis_results, Save_vis_img_path)
