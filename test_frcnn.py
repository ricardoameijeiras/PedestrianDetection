from __future__ import division

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import cv2 
import numpy as np
import sys
import pickle
from matplotlib import pyplot as plt

from optparse import OptionParser
import time

from keras import backend as K
from keras.layers import Input
from keras.models import Model

from faster_rcnn import config
from faster_rcnn import roi_helpers

import faster_rcnn.vgg16 as nn

sys.setrecursionlimit(40000)

# Configuration manually without parser
base_path = '/home/benan/Detektion'

test_path = '/home/benan/Open_Images_Downloader/OID/Dataset/test_an.txt'
test_base_path = '//home/benan/Open_Images_Downloader/OID/Dataset/test' # Changed for car detection

# config_output_filename = options.config_filename
config_output_filename = os.path.join(base_path, 'config.pickle')

with open(config_output_filename, 'rb') as f_in:
	C = pickle.load(f_in)

# Flags for Data Augmentation
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

def format_img_size(img, C):
	""" Formats image size based on configuration """
	img_min_side = float(C.im_size)
	(height, width, _) = img.shape
		
	if width <= height:
		ratio = img_min_side / width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side / height
		new_width = int(ratio * width)
		new_height = int(img_min_side)

	img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_CUBIC)

	return img, ratio	

def format_img_channels(img, C):
	""" Formats image channels based on configuration """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)

	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]

	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis = 0)

	return img

def format_img(img, C):
	""" Formats image for model prediction based on configuration """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)

	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2, real_y2)

C.num_rois = 4 # int(num_rois)
num_features = 512

input_shape_img = (None, None, 3)
input_shape_features = (None, None, num_features)

img_input = Input(shape = input_shape_img)
roi_input = Input(shape = (C.num_rois, 4))
feature_map_input = Input(shape = input_shape_features)

# Define base network (VGG here)
shared_layers = nn.nn_base(img_input, trainable = True)

# Define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn_layer(shared_layers, num_anchors)

classifier = nn.classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes = len(C.class_mapping))

# Defining models
model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)
model_classifier = Model([feature_map_input, roi_input], classifier)

# Loading weights
model_rpn.load_weights(C.model_path, by_name = True)
model_classifier.load_weights(C.model_path, by_name = True)

# Compiling models
model_rpn.compile(optimizer = 'sgd', loss = 'mse')
model_classifier.compile(optimizer = 'sgd', loss = 'mse')


# Switch key value for class mapping
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

test_imgs = os.listdir(test_base_path)

imgs_path = []

for i in range(12):
	idx = np.random.randint(len(test_imgs))
	imgs_path.append(test_imgs[idx])

all_imgs = []

classes = {}
bbox_threshold = 0.6

# Loading weights
print('Loading weights from {}'.format(C.model_path))

visualize = True

for idx, img_name in enumerate(sorted(imgs_path)):

	if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.JPG')):
		continue

	print(img_name)

	st = time.time()
	filepath = os.path.join(test_base_path, img_name)

	img = cv2.imread(filepath)

	X, ratio = format_img(img, C)
	X = np.transpose(X, (0, 2, 3, 1))

	# Getting feature-maps F & output layer Y1, Y2 from  RPN
	[Y1, Y2, F] = model_rpn.predict(X)
	
	# Getting boxes by applying NMS -> R.shape = (300, 4)
	R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh = 0.7)

	# Conversion from (x1,x2,y1,y2) to (x,y,w,h)
	R[:, 2] -= R[:, 0]
	R[:, 3] -= R[:, 1]

	# Spatial pyramid pooling to proposed regions
	bboxes = {}
	probs = {}

	for jk in range(R.shape[0] // (C.num_rois + 1)):
		ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis = 0)
		if ROIs.shape[1] == 0:
			break

		if jk == R.shape[0]//C.num_rois:
			
			curr_shape = ROIs.shape
			target_shape = (curr_shape[0], C.num_rois, curr_shape[2])

			ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
			ROIs_padded[:, :curr_shape[1], :] = ROIs
			ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
			ROIs = ROIs_padded

		[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

		for ii in range(P_cls.shape[1]):

			if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
				continue

			cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
			
			if cls_name not in bboxes:
				bboxes[cls_name] = [] #
				probs[cls_name] = [] #

			(x, y, w, h) = ROIs[0, ii, :]

			cls_num = np.argmax(P_cls[0, ii, :])

			try:
				(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
				tx /= C.classifier_regr_std[0]
				ty /= C.classifier_regr_std[1]
				tw /= C.classifier_regr_std[2]
				th /= C.classifier_regr_std[3]

				x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)

			except:
				pass
			
	
			bboxes[cls_name].append([C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)]) #
			probs[cls_name].append(np.max(P_cls[0, ii, :]))

	all_dets = []

	for key in bboxes:
		if key == 'Pedestrian':
			bbox = np.array(bboxes[key])

			new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh = 0.2)
			for jk in range(new_boxes.shape[0]):
				(x1, y1, x2, y2) = new_boxes[jk,:]

				(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

				cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

				textLabel = '{}: {}'.format('Vehicle', int(100 * new_probs[jk]))
				all_dets.append((key, 100 * new_probs[jk]))

				(retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
				textOrg = (real_x1, real_y1 - 0)

				cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5), (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 1)
				cv2.rectangle(img, (textOrg[0] - 5,textOrg[1] + baseLine - 5), (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)

				cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

		else:
			continue
	
	

	print('Elapsed time = {}'.format(time.time() - st))

	print(all_dets)
	plt.figure(figsize=(10,10))
	plt.grid()
	plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	plt.show()
# cv2.imwrite('./results_imgs/{}.png'.format(idx),img)
