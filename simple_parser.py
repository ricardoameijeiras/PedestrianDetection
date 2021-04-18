import cv2
import numpy as np

# Input-path as path to annotation text-file (train & validation)
def get_data(input_path):
	found_bg = False

	all_imgs = {}
	classes_count = {}
	class_mapping = {}

	visualize = True
	
	with open(input_path,'r') as f:

		print('Parsing annotation files')

		for line in f:
			line_split = line.strip().split(',')
			(filename, class_name, x1, y1, x2, y2) = line_split # Format of annotation file
			
			# Image path-name
			filename = '/home/benan/Open_Images_Downloader/OID/Dataset/train/' + filename + '.jpg'

			# Calculating images for each class (class_name) ...
			if class_name not in classes_count:
				classes_count[class_name] = 1
			else:
				classes_count[class_name] += 1

			if class_name not in class_mapping:

				if class_name == 'bg' and found_bg == False:
					print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
					found_bg = True

				class_mapping[class_name] = len(class_mapping)

			if filename not in all_imgs:
				all_imgs[filename] = {}
				
				img = cv2.imread(filename)
				(rows, cols) = img.shape[:2]

				all_imgs[filename]['filepath'] = filename
				all_imgs[filename]['width'] = cols
				all_imgs[filename]['height'] = rows
				all_imgs[filename]['bboxes'] = []

				
				# Training & validation images -> splitting up data with 80/20 ratio
				if np.random.randint(0, 5) > 0:
					all_imgs[filename]['imageset'] = 'trainval' # Training

				else:
					all_imgs[filename]['imageset'] = 'test' # Validation

			all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': round(float(x1)), 'x2': round(float(x2)), 'y1': round(float(y1)), 'y2': round(float(y2))})


		all_data = []

		for key in all_imgs:
			all_data.append(all_imgs[key])
		
		# Background class is last in list ...
		if found_bg:

			if class_mapping['bg'] != len(class_mapping) - 1:
				key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
				val_to_switch = class_mapping['bg']
				class_mapping['bg'] = len(class_mapping) - 1
				class_mapping[key_to_switch] = val_to_switch
		
	return all_data, classes_count, class_mapping
