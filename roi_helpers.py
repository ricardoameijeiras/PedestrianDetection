# -*- coding: utf-8 -*-

import numpy as np
import pdb
import math

import data_augment
import data_generators

import copy
import time


def calc_iou(R, img_data, C, class_mapping):

	bboxes = img_data['bboxes']
	(width, height) = (img_data['width'], img_data['height'])
	
	# Resizing v. Image -> Mindestgröße kleinerer Kante (px = 600)!
	(resized_width, resized_height) = data_augment.get_new_img_size(width, height, C.im_size)

	# Initialisierung GT Box-Matrix mit Koordinaten
	gta = np.zeros((len(bboxes), 4))

	for bbox_num, bbox in enumerate(bboxes):

		# Abbildung Eck-Koordinaten GT Box auf Feature-Map -> Resize & Stride (r = 16) 
		gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width)) / C.rpn_stride))   # x-Koordinate 
		gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / C.rpn_stride))   # x-Koordinate 
		gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / C.rpn_stride)) # y-Koordinate 
		gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / C.rpn_stride)) # y-Koordinate 

	# Initialisierung Listen
	x_roi = []
	y_class_num = []
	y_class_regr_coords = []
	y_class_regr_label = []
	IoUs = []

	# Für jede ROI (Region-Proposal) im Image ...
	for ix in range(R.shape[0]):

		# Initialisierung Eck-Koordinaten mit Inhalt ROI-Matrix
		(x1, y1, x2, y2) = R[ix, :]

		x1 = int(round(x1)) 
		y1 = int(round(y1))
		x2 = int(round(x2))
		y2 = int(round(y2))

		best_iou = 0.0
		best_bbox = -1

		# Für jede Ground-Truth Box im Image ...
		for bbox_num in range(len(bboxes)):

			# IOU (Ground-Truth Box & ROI im Image)
			curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])

			# Falls IOU v. GT Box & ROI größer als momentane IOU ist ...
			if curr_iou > best_iou:

				best_iou = curr_iou  # Initialisierung Variable mit Wert
				best_bbox = bbox_num 

		# Falls IOU v. ROI mit Ground-Truth Box kleiner als Mindestgrenzwert ist -> nächste ROI
		if best_iou < C.classifier_min_overlap:
				continue

		else:
			w = x2 - x1 # ROI-Weite
			h = y2 - y1 # ROI-Höhe

			x_roi.append([x1, y1, w, h]) # Einfügen ROI-Werte (Ecke, Weite, Höhe) in ROI-Liste
			IoUs.append(best_iou)	     # Einfügen IOU v. ROI mit GT Box in IOU-Liste

			# Falls IOU < 0.5 & IOU >= 0.1 ist ...
			if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
				cls_name = 'bg'

			# Falls IOU >= 0.5 ist ...
			elif C.classifier_max_overlap <= best_iou:
				cls_name = bboxes[best_bbox]['class']		    # ROI erhält Klassen-Name v. GT Box

				cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0 # x-Koordinate Zentrum GT Box
				cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0 # y-Koordinate Zentrum GT Box

				cx = x1 + w / 2.0 # x-Koordinate Zentrum ROI
				cy = y1 + h / 2.0 # y-Koordinate Zentrum ROI

				# Regressions-Variablen für Regressions-Kostenfunktion
				tx = (cxg - cx) / float(w)
				ty = (cyg - cy) / float(h)
				tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
				th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))

			else:
				print('roi = {}'.format(best_iou))
				raise RuntimeError

		class_num = class_mapping[cls_name]	       # Klassen-Nummer         -> Abbildung Klassen-Name
		class_label = len(class_mapping) * [0]	       # 0-Vektor (Klasse)      -> Länge Klassen-Anzahl
		class_label[class_num] = 1		       # Vektor wird an Klassen-Nummer auf 1 gesetzt	
	
		y_class_num.append(copy.deepcopy(class_label)) # Einfügen Vektor in Klassen-Liste

		coords = [0] * 4 * (len(class_mapping) - 1)    # 0-Vektor (Regressions-Variable)
		labels = [0] * 4 * (len(class_mapping) - 1)    # 0-Vektor (Labels) -> jede Klasse 4 Labels

		# Falls ROI kein Background darstellt ...
		if cls_name != 'bg':
			label_pos = 4 * class_num	       # Einstellung Label-Position				
			sx, sy, sw, sh = C.classifier_regr_std 

			coords[label_pos:(4 + label_pos)] = [(sx * tx), (sy * ty), (sw * tw), (sh * th)] # Visualisierung Regressions-Variable in Vektor -> Label-Position
			labels[label_pos:(4 + label_pos)] = [1, 1, 1, 1]				 # Visualisierung Label in Vektor		 -> Label-Position

			y_class_regr_coords.append(copy.deepcopy(coords)) # Einfügen Regressions-Vektor in Liste
			y_class_regr_label.append(copy.deepcopy(labels))  # Einfügen Label-Vektor in Liste
		
		# Falls ROI Background darstellt ...	
		else:
			y_class_regr_coords.append(copy.deepcopy(coords)) 
			y_class_regr_label.append(copy.deepcopy(labels))

	# Falls ROI-Liste keine Elemente enthält ...
	if len(x_roi) == 0:
		return None, None, None, None

	# Erstellung Arrays aus ROI-Liste & Klassen-Liste
	X = np.array(x_roi)
	Y1 = np.array(y_class_num)

	Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis = 1)

	return np.expand_dims(X, axis = 0), np.expand_dims(Y1, axis = 0), np.expand_dims(Y2, axis = 0), IoUs


def apply_regr(x, y, w, h, tx, ty, tw, th):

	try:
		cx = x + (w / 2.)
		cy = y + (h / 2.)
		cx1 = tx * w + cx
		cy1 = ty * h + cy

		w1 = math.exp(tw) * w
		h1 = math.exp(th) * h
		x1 = cx1 - (w1 / 2.)
		y1 = cy1 - (h1 / 2.)

		x1 = int(round(x1))
		y1 = int(round(y1))
		w1 = int(round(w1))
		h1 = int(round(h1))

		return x1, y1, w1, h1

	except ValueError:
		return x, y, w, h

	except OverflowError:
		return x, y, w, h

	except Exception as e:
		print(e)

		return x, y, w, h


def apply_regr_np(X, T):

	try:
		x = X[0, :, :]
		y = X[1, :, :]
		w = X[2, :, :]
		h = X[3, :, :]

		tx = T[0, :, :]
		ty = T[1, :, :]
		tw = T[2, :, :]
		th = T[3, :, :]

		cx = x + (w / 2.)
		cy = y + (h / 2.)
		cx1 = (tx * w) + cx
		cy1 = (ty * h) + cy

		w1 = np.exp(tw.astype(np.float64)) * w
		h1 = np.exp(th.astype(np.float64)) * h
		x1 = cx1 - w1/2.
		y1 = cy1 - h1/2.

		x1 = np.round(x1)
		y1 = np.round(y1)
		w1 = np.round(w1)
		h1 = np.round(h1)

		return np.stack([x1, y1, w1, h1])

	except Exception as e:
		print(e)

		return X

# Funktion NMS -> Verringerung ROI-Zahl
def non_max_suppression_fast(boxes, probs, overlap_thresh = 0.9, max_boxes = 300):

	# Falls keine Region-Proposals vorhanden sind -> Rückgabe leerer Liste
	if len(boxes) == 0:
		return []

	# Initialisierung Eck-Koordinaten 
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	np.testing.assert_array_less(x1, x2)
	np.testing.assert_array_less(y1, y2)

	# Falls ROI-Werte als Integer visualisiert werden ...
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# Initialisierung Auswahl-Liste	
	pick = []

	# Flächen-Berechnung
	area = (x2 - x1) * (y2 - y1)

	# Sortierung ROI -> Wahrscheinlichkeit für Objekt (Index-Liste)
	idxs = np.argsort(probs)

	# Falls Index-Liste mindestens 1 Element enthält ...
	while len(idxs) > 0:
	
		last = len(idxs) - 1
		i = idxs[last]
		
		# Letzter Index wird in Auswahl-Liste eingefügt
		pick.append(i)

		# ROI i (last) wird mit restl. ROIs verarbeitet
		xx1_int = np.maximum(x1[i], x1[idxs[:last]]) # Intersection x1 Eck-Koordinate v. Index i (letzter ROI-Index) & restl. ROIs -> ROI 0 - ROI (last-1)
		yy1_int = np.maximum(y1[i], y1[idxs[:last]]) # Intersection y1 Eck-Koordinate v. Index i (letzter ROI-Index) & restl. ROIs
		xx2_int = np.minimum(x2[i], x2[idxs[:last]]) # Intersection x2 Eck-Koordinate v. Index i (letzter ROI-Index) & restl. ROIs
		yy2_int = np.minimum(y2[i], y2[idxs[:last]]) # Intersection y2 Eck-Koordinate v. Index i (letzter ROI-Index) & restl. ROIs

		ww_int = np.maximum(0, xx2_int - xx1_int)    # Intersection-Weite  -> Liste
		hh_int = np.maximum(0, yy2_int - yy1_int)    # Intersection-Höhe   -> Liste

		area_int = ww_int * hh_int		     # Intersection-Fläche -> Liste

		# Union zw. ROI i & restl. ROIs
		area_union = area[i] + area[idxs[:last]] - area_int

		# Einzelne Überlappung v. ROI i & restl. ROIs
		overlap = (area_int / (area_union + 1e-6))

		# Löschen v. Indicies (ROIs) aus Index-Liste mit overlap > 0.9 mit ROI i
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlap_thresh)[0])))

		# Falls Auswahl-Liste größer/gleich 'max_boxes' ist ...
		if len(pick) >= max_boxes:
			break

	# Rückgabe v. Region-Proposals (Anzahl = 'max_boxes') & Objekt-Wahrscheinlichkeiten v. Region-Proposals
	boxes = boxes[pick].astype("int")
	probs = probs[pick]

	return boxes, probs


def rpn_to_roi(rpn_layer, regr_layer, C, use_regr = True, max_boxes = 300, overlap_thresh = 0.9):

	regr_layer = regr_layer / C.std_scaling

	# Konfigurations-Variable
	anchor_sizes = C.anchor_box_scales  # Anchor-Box Skalen
	anchor_ratios = C.anchor_box_ratios # Anchor-Box Verhältnisse

	assert rpn_layer.shape[0] == 1

	# Definition Zeilen & Spalten aus Tensorflow-Backend
	# if dim_ordering == 'tf':
	(rows, cols) = rpn_layer.shape[1:3]

	curr_layer = 0

	# Initialisierung 4D Tensor -> 4 3D Tensoren
	# if dim_ordering == 'tf':
	A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))

	# Für jede Anchor-Box Skala ...
	for anchor_size in anchor_sizes:

		# Für jedes Anchor-Box Verhältnis ...
		for anchor_ratio in anchor_ratios:

			anchor_x = (anchor_size * anchor_ratio[0]) / C.rpn_stride # Abbildung Anchor-Box Weite auf FM
			anchor_y = (anchor_size * anchor_ratio[1]) / C.rpn_stride # Abbildung Anchor-Box Höhe auf FM

			# if dim_ordering == 'tf':
			regr = regr_layer[0, :, :, (4 * curr_layer):(4 * curr_layer) + 4]
			regr = np.transpose(regr, (2, 0, 1))

			# Erstellung Grid-Koordinaten v. Spalten & Zeilen (rows x cols)
			X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

			# Jeder Tensor stellt 3D-Visualisierung v. Ecke (oben links), Höhe, Weite Anchor-Box dar -> 2D-Matrix wird für jeden Layer ausgefüllt!
			A[0, :, :, curr_layer] = X - (anchor_x / 2) # x Eck-Koordinate
			A[1, :, :, curr_layer] = Y - (anchor_y / 2) # y Eck-Koordinate
			A[2, :, :, curr_layer] = anchor_x	    # Anchor-Box Weite
			A[3, :, :, curr_layer] = anchor_y           # Anchor-Box Höhe

			# Regression auf 4 Anchor-Box Tensoren (4 AB Tensoren)
			if use_regr:
				A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

			A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer]) # Anchor-Box Weite mind. 1
			A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer]) # Anchor-Box Höhe mind. 1

			A[2, :, :, curr_layer] += A[0, :, :, curr_layer] # x2 Eck-Koordinate Anchor-Box	       
			A[3, :, :, curr_layer] += A[1, :, :, curr_layer] # y2 Eck-Koordinate Anchor-Box	       

			A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])        # x1 Eck-Koordinate mind. 0
			A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])        # y1 Eck-Koordinate mind. 0
			A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer]) 
			A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])

			# Inkrementierung 
			curr_layer += 1

	# Reshape zu 2D-Matrix & Transponierung (ROI-Matrix) -> (4 x (W * H * 9))
	all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
	# Reshape -> RPN-Layer (1 x 14 x 14 x 9) reshape zu (14 * 14 * 9) Probability-werten
	all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

	# 2D-Matrix als ROI-Matrix mit 4 Spalten
	x1 = all_boxes[:, 0] # x1 Eck-Koordinate
	y1 = all_boxes[:, 1] # x2 Eck-Koordinate
	x2 = all_boxes[:, 2] # y1 Eck-Koordinate
	y2 = all_boxes[:, 3] # y2 Eck-Koordinate

	# Index-Liste für alle ROIs mit Bedingung ...
	idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))

	# Löschen v. ROIs in Index-Liste aus 'all_boxes' & 'all_probs' 
	all_boxes = np.delete(all_boxes, idxs, 0)
	all_probs = np.delete(all_probs, idxs, 0)

	# Anwendung Non-Max Suppression auf ROIs
	result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh = overlap_thresh, max_boxes = max_boxes)[0]

	return result
