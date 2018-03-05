# This script converts image annotations in a val/test into a RLE format json dataset 

import os
import glob
import argparse
import json
import numpy as np
import cv2
from imageio import imread
from pycocotools import mask as COCOmask


def parse_args():
	parser = argparse.ArgumentParser(description='Evaluation demo')
	parser.add_argument('--ann_dir', default='./annotations_instance/validation') # CHANGE ACCORDINGLY
	parser.add_argument('--imgCatIdsFile', default='../imgCatIds.json')
	parser.add_argument('--output_json', default='./instance_validation_gts.json')
	
	args = parser.parse_args()
	return args

def maskToPolygon(mask):
    mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c,h in zip(contours, hierarchy[0]) if h[3] < 0] # Only outermost polygon

    segmentation = []
    for contour in contours:
        polygon = contour.flatten().tolist()
        if len(polygon) >= 6:
            segmentation.append(polygon)
    if len(segmentation) == 0:
        return None
    return segmentation

def convert(args):
	data_dict = json.load(open(args.imgCatIdsFile, 'r'))
	img2id = {x['file_name']: x['id'] 
			  for x in data_dict['images']}
	img2info = {x['file_name']: x 
			    for x in data_dict['images']}

	categories = data_dict['categories'] 
	images = []
	images_unique = set()
	annotations = []
	ann_id = 0
	# loop over annotation files
	files_ann = sorted(glob.glob(os.path.join(args.ann_dir, '*.png')))
	for i, file_ann in enumerate(files_ann):
		if i % 50 == 0:
			print('#files processed: {}'.format(i))

		file_name = os.path.basename(file_ann).replace('.png', '.jpg')
		img_id = img2id[file_name]
		if file_name not in images_unique:
			images_unique.add(file_name)
			images.append(img2info[file_name])
		ann_mask = imread(file_ann)
		Om = ann_mask[:, :, 0]
		Oi = ann_mask[:, :, 1]
		
		# loop over instances
		for instIdx in np.unique(Oi):
			if instIdx == 0:
				continue
			imask = (Oi == instIdx)
			cat_id = Om[imask][0]
			imask = imask[:,:,np.newaxis]
			
			# Detectron expects ground truth to be polygons, not RLE
			polygon = maskToPolygon(imask)
                        if polygon is None:
                            print(file_ann, instIdx)
                            continue
			# RLE encoding
			rle = COCOmask.encode(np.asfortranarray(imask.astype(np.uint8)))
                        bbox = COCOmask.toBbox(rle)
			bbox = bbox[0]
			
                        ann = {}
			ann['id'] = ann_id
			ann_id += 1
			ann['image_id'] = img_id
			ann['segmentation'] = polygon
			ann['bbox'] = bbox.tolist()
			ann['category_id'] = int(cat_id)
			ann['iscrowd'] = 0
			ann['area'] = np.sum(imask)
			annotations.append(ann)
	
	# data_dict['annotations'] = annotations
	print('#files: {}, #instances: {}'.format(len(files_ann), len(annotations)))

	data_out = {'categories': categories, 'images': images, 'annotations': annotations}
	with open(args.output_json, 'w') as f:
		json.dump(data_out, f)


if __name__ == '__main__':
	args = parse_args()
	convert(args)
