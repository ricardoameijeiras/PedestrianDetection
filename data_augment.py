# -*- coding: utf-8 -*-

''' Python-Skript -> Durchführung Daten-Augmentation
    - Erhöhung des Daten-Sets
    - Horizontale, vertikale Drehung & Rotation um 90°
'''

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2
import numpy as np
import copy

# Neue Image-Größe abhängig v. Mindestgröße kürzerer Seite
def get_new_img_size(width, height, img_min_side = 300):

	# Anpassung Weite auf px = 600
	if width <= height:

		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side

	# Anpassung Höhe auf px = 600
	else:

		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height



def augment(img_data, config, augment = True):

	# Assert-Bedingungen
	assert 'filepath' in img_data # Datei-Pfad
	assert 'bboxes' in img_data   # Ground-Truth Bounding-Boxen
	assert 'width' in img_data    # Image-Weite
	assert 'height' in img_data   # Image-Höhe

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2]

		# Horizontale Drehung -> Setting-Variable in Konfigurations-Datei
		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)

			# Für alle Ground-Truth Bounding-Boxen im Image
			for bbox in img_data_aug['bboxes']:

				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		# Vertikale Drehung -> Setting-Variable in Konfigurations-Datei
		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)

			for bbox in img_data_aug['bboxes']:

				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1 # Neue y2-Koordinate
				bbox['y1'] = rows - y2 # Neue y1-Koordinate

		# Rotation -> Rotation d. Image um 0°, 90°, 180°, 270°
		if config.rot_90:

			angle = np.random.choice([0,90,180,270], 1)[0]

			if angle == 0:
				pass

			# Image-Matrix Transponierung & horizontale Drehung -> Rotation um 90°
			elif angle == 90:
				img = np.transpose(img, (1, 0, 2))
				img = cv2.flip(img, 1)

			# Horizontale & vertikale Drehung -> Rotation um 180°
			elif angle == 180:
				img = cv2.flip(img, -1)

			# Image-Matrix Transponierung & vertikale Drehung -> Rotation um 270°
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)

			# Für alle Ground-Truth Bounding-Boxen im Image
			for bbox in img_data_aug['bboxes']:

				x1 = bbox['x1'] # Initialisierung x1-Koordinate
				x2 = bbox['x2'] # Initialisierung x2-Koordinate
				y1 = bbox['y1'] # Initialisierung y1-Koordinate
				y2 = bbox['y2'] # Initialisierung y2-Koordinate

				if angle == 0:
					pass

				# Neue Koordinaten (x1, x2, y1, y2) für Ground-Truth Box im Image 
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2

				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1

				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2

	# Initialisierung Höhe & Weite -> augmentiertes Image
	img_data_aug['height'] = img.shape[0]
	img_data_aug['width'] = img.shape[1]
	
	return img_data_aug, img
