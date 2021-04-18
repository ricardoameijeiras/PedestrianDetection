# -*- coding: utf-8 -*-

from __future__ import absolute_import

#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import data_augment
import numpy as np
import cv2
import random
import copy
import threading
import itertools

# Union v. 2 Flächen
def union(au, bu, area_intersection):

	area_a = (au[2] - au[0]) * (au[3] - au[1]) # Flächeninhalt Fläche A = (x2 - x1) * (y2 - y1)
	area_b = (bu[2] - bu[0]) * (bu[3] - bu[1]) # Flächeninhalt Fläche B = (x2 - x1) * (y2 - y1)
	area_union = area_a + area_b - area_intersection

	return area_union

# Überschneidungsfläche v. 2 Flächen
def intersection(ai, bi):

	x = max(ai[0], bi[0])     # Koordinate x1 als Maximalwert v. Fläche A, Fläche B 
	y = max(ai[1], bi[1])     # Koordinate y1 als Maximalwert v. Fläche A, Fläche B
	w = min(ai[2], bi[2]) - x # Koordinate x2 als Minimalwert v. Fläche A, Fläche B -> Intersection-Weite (x2 - x1)
	h = min(ai[3], bi[3]) - y # Koordinate y2 als Minimalwert v. Fläche A, Fläche B -> Intersection-Weite (y2 - y1)

	if w < 0 or h < 0:
		return 0
	return (w * h)

# IOU (Intersection over Union) v. Fläche A, Fläche B
def iou(a, b):

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0

	area_i = intersection(a, b)  # Überschneidungsfläche
	area_u = union(a, b, area_i) # Union

	return float(area_i) / float(area_u + 1e-6)

'''
class SampleSelector:

	def __init__(self, class_count):

		# Samples werden nach Klassen durchsucht -> Liste & Iteration
		self.classes = [b for b in class_count.keys() if class_count[b] > 0]
		self.class_cycle = itertools.cycle(self.classes)
		self.curr_class = next(self.class_cycle)

	# Entwertung Sample für balancierte Klassen im Daten-Set
	def skip_sample_for_balanced_class(self, img_data):

		class_in_img = False

		# Für jede Ground-Truth Bounding-Box im Image ...
		for bbox in img_data['bboxes']:

			cls_name = bbox['class']

			if cls_name == self.curr_class:
				class_in_img = True
				self.curr_class = next(self.class_cycle)
				break

		if class_in_img:
			return False

		else:
			return True
'''

# Kalkulation d. Resultas d. Region-Proposal Networks
def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):

	downscale = float(C.rpn_stride)     # Stride für VGG16 = 16
	anchor_sizes = C.anchor_box_scales  # Anchor-Größen = [128, 256, 512]
	anchor_ratios = C.anchor_box_ratios # Anchor-Verhältnisse (1:1), (1:2), (2:1)
	num_anchors = len(anchor_sizes) * len(anchor_ratios)	

	# Output-Größe d. Basis-Netzwerks (Feature-Map)
	(output_width, output_height) = img_length_calc_function(resized_width, resized_height)

	n_anchratios = len(anchor_ratios)
	
	# Initialisierung leerer Tensoren für RPN-Objekte
	y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))	# 3D Tensor mit jeweils 9 Elementen (14 x 14 x 9)           -> Überlappungs-Tensor
	y_is_box_valid = np.zeros((output_height, output_width, num_anchors))	# 3D Tensor mit jeweils 9 Elementen (14 x 14 x 9)           -> Label-Tensor
	y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))	# 3D Tensor mit jeweils 36 (4 * 9) Elementen (14 x 14 x 36) -> Regressions-Tensor	

	# Ground-Truth Box-Anzahl im Image
	num_bboxes = len(img_data['bboxes'])

	num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)		   	# Initialisierung horizontaler Vektor
	best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)	# Initialisierung Matrix (num_bboxes x 4) mit 4 Anchor-Box Koordinaten
	best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)		# Initialisierung horizontaler Vektor
	best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)			# Initialisierung Matrix (num_bboxes x 4)
	best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)		# Initialisierung Matrix (num_bboxes x 4)

	# Ground-Truth Box-Koordinaten -> nach Resizing auf px = 600 (Mindestgröße)
	gta = np.zeros((num_bboxes, 4))

	for bbox_num, bbox in enumerate(img_data['bboxes']):

		# Ground-Truth Box-Koordinaten
		gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
		gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
		gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
		gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))
	
	# Für jede Anchor-Box ...
	for anchor_size_idx in range(len(anchor_sizes)):
		for anchor_ratio_idx in range(n_anchratios):

			anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0] # Anchor-Box Weite
			anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1] # Anchor-Box Höhe	
			
			for ix in range(output_width):					
				# x-Koordinaten für Anchor-Box auf Feature-Map -> Zentrum in Pixel ix!	
				x1_anc = downscale * (ix + 0.5) - anchor_x / 2
				x2_anc = downscale * (ix + 0.5) + anchor_x / 2	
				
				# Ausschluss v. Anchor-Box, falls dieser über die Grenze d. Image-Weite geht					
				if x1_anc < 0 or x2_anc > resized_width:
					continue
					
				for jy in range(output_height):

					# y-Koordinaten für Anchor-Box auf Feature-Map -> Zentrum in Pixel jy!
					y1_anc = downscale * (jy + 0.5) - anchor_y / 2
					y2_anc = downscale * (jy + 0.5) + anchor_y / 2

					# Ausschluss v. Anchor-Box, falls dieser über die Grenze d. Image-Höhe geht
					if y1_anc < 0 or y2_anc > resized_height:
						continue

					# Default-Label für Anchor-Box 
					bbox_type = 'neg'

					###
					best_iou_for_loc = 0.0

					# Für jede Ground-Truth Box im Image
					for bbox_num in range(num_bboxes):
						
						# IOU d. Ground-Truth Box & d. jeweiligen Anchor-Box
						curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1_anc, y1_anc, x2_anc, y2_anc])

						# Regression-Targets werden visualisiert (tx, ty, tw, th) -> positive Anchor-Box mit IOU > 0.7 (mit GT Box)
						if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
							cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0 # x-Koordinate Zentrum GT Box
							cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0 # y-Koordinate Zentrum GT Box
							cxa = (x1_anc + x2_anc) / 2.0			 # x-Koordinate Zentrum Anchor-Box
							cya = (y1_anc + y2_anc) / 2.0			 # y-Koordinate Zentrum Anchor-Box

							tx = (cx - cxa) / (x2_anc - x1_anc)
							ty = (cy - cya) / (y2_anc - y1_anc)
							tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
							th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))
						
						# Falls Ground-Truth Box im Image keine Hintergrund-Klasse darstellt ...
						if img_data['bboxes'][bbox_num]['class'] != 'bg':

							# Alle GT Boxen im Image werden mit Anchor-Box abgeglichen -> neue Überlappung wird gesetzt
							if curr_iou > best_iou_for_bbox[bbox_num]:

								best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx] # Anchor-Box Grid-Koordinaten 
								best_iou_for_bbox[bbox_num] = curr_iou					     # Überlappung AB mit GTB
								best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]		     # Anchor-Box Koordinaten (FM)
								best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]			     # Regressions-Variablen

							# Positive Anchor-Box wird gesetzt für IOU > 0.7 mit GT Box
							if curr_iou > C.rpn_max_overlap:

								bbox_type = 'pos'
								num_anchors_for_bbox[bbox_num] += 1 # Anchor-Box Anzahl inkrementiert für GTB
								
								###
								if curr_iou > best_iou_for_loc:
									best_iou_for_loc = curr_iou
									best_regr = (tx, ty, tw, th)

							# Neutrale Anchor-Box für IOU > 0.3 mit GT Box & IOU < 0.7 mit GT Box
							if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
								
								if bbox_type != 'pos':
									bbox_type = 'neutral'

					# Einstellung initialisierter Outputs -> Abhängigkeit v. Anchor-Box Typ (positiv, neutral, negativ)!
					if bbox_type == 'neg':

						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1 # Label-Tensor an Grid-Pixel (ix, jy) auf 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0  # Überlappungs-Tensor an Grid-Pixel (ix, jy) auf 0

					elif bbox_type == 'neutral':

						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0 # Label-Tensor an Grid-Pixel (ix, jy) auf 0
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0  # Überlappungs-Tensor an Grid-Pixel (ix, jy) auf 0

					elif bbox_type == 'pos':

						y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1 # Label-Tensor an Grid-Pixel (ix, jy) auf 1
						y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1  # Überlappungs-Tensor an Grid-Pixel (ix, jy) auf 1

						start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx) # Höchstwert = 8 -> Startwert = 32
						y_rpn_regr[jy, ix, start:(start + 4)] = best_regr 		# Regressions-Tensor an Grid-Pixel (ix, jy) mit Regressions-Werten (tx, ty, tw, th) versehen

	
	# Jede GT Bounding-Box muss mind. 1 positive RPN-Region aufweisen
	for idx in range(num_anchors_for_bbox.shape[0]):

		if num_anchors_for_bbox[idx] == 0:

			# Falls für eine GT Box keine positiven Anchor-Boxen gefunden werden ...
			if best_anchor_for_bbox[idx, 0] == -1:
				continue

			# Label-Tensor & Überlappungs-Tensor erhält an Grid-Koordinate (jy, ix) Wert 1
			y_is_box_valid[
				best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[idx, 2] + n_anchratios *
				best_anchor_for_bbox[idx, 3]] = 1
			y_rpn_overlap[
				best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[idx, 2] + n_anchratios *
				best_anchor_for_bbox[idx, 3]] = 1

			# Regreessions-Tensor wird mit Regressions-Variablen versehen
			start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
			y_rpn_regr[
				best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:(start + 4)] = best_dx_for_bbox[idx, :]

	# Tensoren-Bearbeitung -> Transponierung & Dimensions-Erhöhung
	y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
	y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis = 0)

	y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
	y_is_box_valid = np.expand_dims(y_is_box_valid, axis = 0)

	y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
	y_rpn_regr = np.expand_dims(y_rpn_regr, axis = 0)

	# Tensor mit True, False Beschriftung, falls für Überlappung & Label 1 gilt -> np.logical_and
	# Rückgabe v. Tupel-Liste v. Indicies v. True-Beschriftungen im Tensor
	pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1)) # Positive Lokalisierungen v. Anchor-Boxen -> positive AB
	neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1)) # Negative Lokalisierungen v. Anchor-Boxen -> negative AB

	# Anzahl v. positiven Anchor-Boxen
	num_pos = len(pos_locs[0])

	# Mini-Batch im Image für Training enthält 256 Anchor-Boxen (1:1 Verhältnis) -> Verhältnis von positiven & negativen Anchor-Boxen
	num_regions = 256

	# Anzahl positiver Anchor-Boxen höher als 128 (256/2) ...
	if len(pos_locs[0]) > num_regions/2:
		val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions/2)	   # Auswahl zufälliger Zahlen (0 - Anzahl pos. AB)
		y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0 # Label-Tensor wird an ausgewählten Stellen auf 0 gesetzt
		num_pos = num_regions/2

	# Anzahl positiver & negativer Anchor-Boxen höher als 256 ...
	if len(neg_locs[0]) + num_pos > num_regions:
		val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)		   # Auswahl zufälliger Zahlen (0 - Anzahl neg. AB)
		y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0 # Label-Tensor wird an ausgewählten Stellen auf 0 gesetzt

	y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis = 1)				   # Tensoren für Überlappung & Labeling werden verkettet   -> Klassifikations-Tensor!
	y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis = 1), y_rpn_regr], axis = 1)	   # Tensoren für Überlappung & Regression werden verkettet -> Regressions-Tensor!

	return np.copy(y_rpn_cls), np.copy(y_rpn_regr), num_pos


'''
class threadsafe_iter:

	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return next(self.it)		

	
def threadsafe_generator(f):

	def g(*a, **kw):
		return threadsafe_iter(f(*a, **kw))
	return g
'''


def get_anchor_gt(all_img_data, C, img_length_calc_function, mode = 'train'):

	# sample_selector = SampleSelector(class_count)

	# Mischen v. Image-Daten im Trainings-Modus
	while True:
		# if mode == 'train':
			# np.random.shuffle(all_img_data)

		# Für jedes Image im Daten-Set ...
		for img_data in all_img_data:
			try:
				# Überspringen v. Image-Daten ...
				# if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
					# continue

				# Lesen & Augmentation v. Image-Datei im Trainings-Modus
				if mode == 'train':
					img_data_aug, x_img = data_augment.augment(img_data, C, augment = True)

				# Lesen v. Image-Datei
				else:
					img_data_aug, x_img = data_augment.augment(img_data, C, augment = False)

				(width, height) = (img_data_aug['width'], img_data_aug['height']) # Höhe & Weite nach Augmentation
				(rows, cols, _) = x_img.shape

				assert cols == width
				assert rows == height

				# Image-Dimensionen nach Resize-Anwendung
				(resized_width, resized_height) = data_augment.get_new_img_size(width, height, C.im_size)

				# Resizing v. Image abhängig v. Image-Dimensionen
				x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation = cv2.INTER_CUBIC)
				debug_img = x_img.copy()

				try:
					# Berechnung Klassifikations-Tensor & Regressions-Tensor d. jeweiligen Images
					y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)

				except:
					continue

				# Zero-center by mean pixel, and preprocess image

				x_img = x_img[:, :, (2, 1, 0)]  # Umstrukturierung Channel-Order -> BGR to RGB
				x_img = x_img.astype(np.float32)

				x_img[:, :, 0] -= C.img_channel_mean[0] # Standardization red image pixel
				x_img[:, :, 1] -= C.img_channel_mean[1] # Standardization yellow image pixel
				x_img[:, :, 2] -= C.img_channel_mean[2] # Standardization green image pixel

				# Skalierung mit Konfigurations-Variable
				x_img /= C.img_scaling_factor

				# Image-Transponierung
				x_img = np.transpose(x_img, (2, 0, 1))
				x_img = np.expand_dims(x_img, axis = 0)

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				# Tensorflow-Backend erfordert Channel-Size als letzte Dimension
				x_img = np.transpose(x_img, (0, 2, 3, 1))
				y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
				y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_img, num_pos

			except Exception as e:

				print(e)
				continue


