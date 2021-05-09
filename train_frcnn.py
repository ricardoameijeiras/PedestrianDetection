from __future__ import division

import os
import random
import pprint
import sys
import time
import numpy as np
import tensorflow as tf
#import pandas as pd
import pickle
import matplotlib.pyplot as plt

from optparse import OptionParser

from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
import config, data_generators
import losses as losses
import roi_helpers
from tensorflow.keras.utils import Progbar

import simple_parser# Simple parser 
import vgg16 as nn	       # VGG base-network

# Should we use it?
sys.setrecursionlimit(40000)

# Configuration manually without parser (here: train mode)
base_path = '/home/benan/Detektion2'
train_path = '/home/benan/Open_Images_Downloader/OID/Dataset/train_an.txt'

# Number of ROIs processed at once
num_rois = 4 

# entation flag
horizontal_flips = True # Augment with horizontal flips in training. 
vertical_flips = True   # Augment with vertical flips in training. 
rot_90 = True           # Augment with 90 degree rotations in training. 

output_weight_path = os.path.join(base_path, 'Model/model_frcnn_vgg.hdf5') # Output weights

#record_path = os.path.join(base_path, 'Model/record.csv')
#record_path2 = os.path.join(base_path, 'Model/record2.csv')

base_weight_path = os.path.join(base_path, 'Model/vgg16_weights.h5')	   # Pretrained weights
config_output_filename = '/home/benan/Detektion2/config.pickle'

# Passing settings & configure configuration objects
C = config.Config()

C.use_horizontal_flips = horizontal_flips
C.use_vertical_flips = vertical_flips
C.rot_90 = rot_90

C.network = 'vgg'

#C.record_path = record_path

C.model_path = output_weight_path
C.num_rois = num_rois
C.base_net_weights = base_weight_path 

st = time.time()
all_imgs, classes_count, class_mapping = simple_parser.get_data(train_path)

if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping
# inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))
print(class_mapping)

# config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C, config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))

# Shuffling images
random.seed(1)
random.shuffle(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

num_imgs = len(all_imgs)

## Adjustment according to mode (manual configuration) ##
# Train data generator -> generating X, Y, image data
print('Num train samples {}'.format(len(train_imgs)))
print('Num validation samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, C, nn.get_img_output_length, mode = 'train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, C, nn.get_img_output_length, mode = 'val')

input_shape_img = (None, None, 3)

img_input = Input(shape = input_shape_img)
roi_input = Input(shape = (None, 4))

# Define base network (VGG here ...)
shared_layers = nn.nn_base(img_input, trainable = True)

# Define RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn_layer(shared_layers, num_anchors)

classifier = nn.classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes = len(classes_count))

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# Model holds RPN & classifier -> load/save weights for models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

## Changed
if not os.path.isfile(C.model_path):
    # Loading pretrained weights -> beginning of training
	try:
		print('This is the first time of your training')
		print('loading weights from {}'.format(C.base_net_weights))
		model_rpn.load_weights(C.base_net_weights, by_name = True)
		model_classifier.load_weights(C.base_net_weights, by_name = True)
	except:
		print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')
    
	# Create the record.csv file to record losses, acc and mAP -> training process
	#record_df = pd.DataFrame(columns = ['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])

	# Create the record2.csv file to record losses, acc and mAP -> validation process
	#record2_df = pd.DataFrame(columns = ['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])

else:
	# Loading already trained model -> continue of training
	print('Continue training based on previous trained model')
	print('Loading weights from {}'.format(C.model_path))
	model_rpn.load_weights(C.model_path, by_name = True)
	model_classifier.load_weights(C.model_path, by_name = True)
    
	# Loading records
	#record_df = pd.read_csv(record_path)
	#record2_df = pd.read_csv(record_path2)

	# Training values
	#r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
	#r_class_acc = record_df['class_acc']
	#r_loss_rpn_cls = record_df['loss_rpn_cls']
	#r_loss_rpn_regr = record_df['loss_rpn_regr']
	#r_loss_class_cls = record_df['loss_class_cls']
	#r_loss_class_regr = record_df['loss_class_regr']
	#r_curr_loss = record_df['curr_loss']
	#r_elapsed_time = record_df['elapsed_time']
	#r_mAP = record_df['mAP']

	# Validation values
	#r_mean_overlapping_bboxes2 = record2_df['mean_overlapping_bboxes']
	#r_class_acc2 = record2_df['class_acc']
	#r_loss_rpn_cls2 = record2_df['loss_rpn_cls']
	#r_loss_rpn_regr2 = record2_df['loss_rpn_regr']
	#r_loss_class_cls2 = record2_df['loss_class_cls']
	#r_loss_class_regr2 = record2_df['loss_class_regr']
	#r_curr_loss2 = record2_df['curr_loss']
	#r_elapsed_time2 = record2_df['elapsed_time']
	#r_mAP2 = record2_df['mAP']

	# print('Already train %dK batches' % (len(record_df)))

# Learning rates for RPN & classifier
optimizer = tf.keras.optimizers.Adam(lr = 1e-5)
optimizer_classifier = tf.keras.optimizers.Adam(lr = 1e-5)
#optimizer_all = tf.keras.optimizers.SGD(lr = 1e-5)

model_rpn.compile(optimizer = optimizer, loss = [losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer = optimizer_classifier, loss = [losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)], metrics = {'dense_class_{}'.format(len(classes_count)): 'accuracy'})

#model_all.compile(optimizer = optimizer_all, loss = 'mae')
model_all.compile(optimizer = 'sgd', loss = 'mae')

#total_epochs = len(record_df)
#r_epochs = len(record_df)

#total_epochs2 = len(record2_df)
#r_epochs2 = len(record2_df)
##

##
epoch_length = 1000                            # Iterations = (Dataset / Batch-Size = 1)
num_epochs = 65    	       		       # Epochs
iter_num = 0	   	       		       # Index -> iteration


# Losses of training process & validation process -> 5 training images & 5 val images!!!
losses = np.zeros((1000, 5))
losses2 = np.zeros((len(val_imgs), 5))

rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []

rpn_accuracy_rpn_monitor_v = []
rpn_accuracy_for_epoch_v = []

# Training
#if len(record_df) == 0:
best_loss = np.Inf

#else:
	#best_loss = np.min(r_curr_loss)


# Validation
#if len(record2_df) == 0:
best_loss2 = np.Inf

#else:
	#best_loss2 = np.min(r_curr_loss2)	



start_time = time.time()

class_mapping_inv = {v: k for k, v in class_mapping.items()}

# Start training process
print('Starting training')
start_time = time.time()

for epoch_num in range(num_epochs):
	
	progbar = Progbar(epoch_length)
	print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

	iter_num_aux = 0
	iter_num = 0

	i = True
	training = True
	validate = False
	
	#while(i):
		try:
			if(training):
				if len(rpn_accuracy_rpn_monitor) == len(train_imgs) and C.verbose:
					mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
					rpn_accuracy_rpn_monitor = []

					print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, len(train_imgs) - 1))

					if mean_overlapping_bboxes == 0:
						print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
					### Stehen lassen

				### Training ###
				# Generate X (x_img) & Label Y ([y_rpn_cls, y_rpn_regr]) 
				X, Y, img_data, debug_img, debug_num_pos = next(data_gen_train)

				# Train RPN model & get loss-value
				loss_rpn = model_rpn.train_on_batch(X, Y)

				# Get predicted RPN from RPN model
				P_rpn = model_rpn.predict_on_batch(X)

				# Conversion of RPN layer to ROI boxes
				R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, use_regr = True, overlap_thresh = 0.7, max_boxes = 300)
				
				# Function calc_iou converts from (x1, y1, x2, y2) to (x, y, w, h)
				X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

				# If there are no matching boxes
				if X2 is None:
					rpn_accuracy_rpn_monitor.append(0)
					rpn_accuracy_for_epoch.append(0)
					continue

				# Positive & negative anchors
				neg_samples = np.where(Y1[0, :, -1] == 1)
				pos_samples = np.where(Y1[0, :, -1] == 0)

				if len(neg_samples) > 0:
					neg_samples = neg_samples[0]
				else:
					neg_samples = []

				if len(pos_samples) > 0:
					pos_samples = pos_samples[0]
				else:
					pos_samples = []
				
				rpn_accuracy_rpn_monitor.append(len(pos_samples))
				rpn_accuracy_for_epoch.append((len(pos_samples)))

				if C.num_rois > 1:
					# If number of positive anchors is larger than 2
					if len(pos_samples) < (C.num_rois // 2):
						selected_pos_samples = pos_samples.tolist()
					else:
						selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace = False).tolist()

					# Randomly choose negative samples (num_rois - num_pos)
					try:
							selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace = False).tolist()
					except:
						selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace = True).tolist()

					# Saving of all positive & negative samples in sel_samples
					sel_samples = selected_pos_samples + selected_neg_samples

				else:
					# For num_rois = 1 -> picking one positive or negative example
					selected_pos_samples = pos_samples.tolist()
					selected_neg_samples = neg_samples.tolist()

					if np.random.randint(0, 2):
						sel_samples = random.choice(neg_samples)
					else:
						sel_samples = random.choice(pos_samples)

				loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

				# Losses of training process
				losses[iter_num, 0] = loss_rpn[1]
				losses[iter_num, 1] = loss_rpn[2]

				losses[iter_num, 2] = loss_class[1]
				losses[iter_num, 3] = loss_class[2]
				losses[iter_num, 4] = loss_class[3]
			
				iter_num += 1

				progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])), ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 3]))])
				
			
				## Ending training process -> with augmentation
				if iter_num == 1000:
					loss_rpn_cls = np.mean(losses[:, 0])
					loss_rpn_regr = np.mean(losses[:, 1])
					loss_class_cls = np.mean(losses[:, 2])
					loss_class_regr = np.mean(losses[:, 3])
					class_acc = np.mean(losses[:, 4])

					mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch) 
					rpn_accuracy_for_epoch = []

					if C.verbose:
						print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
						print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
						print('Loss RPN classifier: {}'.format(loss_rpn_cls))
						print('Loss RPN regression: {}'.format(loss_rpn_regr))
						print('Loss Detector classifier: {}'.format(loss_class_cls))
						print('Loss Detector regression: {}'.format(loss_class_regr))
						print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
						print('Elapsed time: {}'.format(time.time() - start_time))

						elapsed_time = (time.time() - start_time) / 60

					curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
					
					start_time = time.time()

					if curr_loss < best_loss:

						if C.verbose:
							print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))

						best_loss = curr_loss
						model_all.save(C.model_path)

					new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
											'class_acc':round(class_acc, 3), 
											'loss_rpn_cls':round(loss_rpn_cls, 3), 
											'loss_rpn_regr':round(loss_rpn_regr, 3), 
											'loss_class_cls':round(loss_class_cls, 3), 
											'loss_class_regr':round(loss_class_regr, 3), 
											'curr_loss':round(curr_loss, 3), 
											'elapsed_time':round(elapsed_time, 3), 
											'mAP': 0}

					#record_df = record_df.append(new_row, ignore_index = True)
					#record_df.to_csv(record_path, index = 0)
				
					if(epoch_num % 5 == 0):
						training = False
						validate = True
					else: 
						i = False


			if (validate):

				# Generate X (x_img) & Label Y ([y_rpn_cls, y_rpn_regr]) 
				X_v, Y_v, img_data_v, debug_img_v, debug_num_pos_v = next(data_gen_val)

				# Train RPN model & get loss-value
				loss_rpn_v = model_rpn.test_on_batch(X_v, Y_v)

				# Get predicted RPN from RPN model
				P_rpn_v = model_rpn.predict_on_batch(X_v)

				# Conversion of RPN layer to ROI boxes
				R_v = roi_helpers.rpn_to_roi(P_rpn_v[0], P_rpn_v[1], C, use_regr = True, overlap_thresh = 0.7, max_boxes = 300)
			
				# Function calc_iou converts from (x1, y1, x2, y2) to (x, y, w, h)
				X2_v, Y1_v, Y2_v, IouS_v = roi_helpers.calc_iou(R_v, img_data_v, C, class_mapping)

				# If there are no matching boxes
				if X2_v is None:
					rpn_accuracy_rpn_monitor.append(0)
					rpn_accuracy_for_epoch.append(0)
					continue

				# Positive & negative anchors
				neg_samples = np.where(Y1_v[0, :, -1] == 1)
				pos_samples = np.where(Y1_v[0, :, -1] == 0)

				if len(neg_samples) > 0:
					neg_samples = neg_samples[0]
				else:
					neg_samples = []

				if len(pos_samples) > 0:
					pos_samples = pos_samples[0]
				else:
					pos_samples = []
		
				rpn_accuracy_rpn_monitor.append(len(pos_samples))
				rpn_accuracy_for_epoch.append((len(pos_samples)))

				if C.num_rois > 1:
					# If number of positive anchors is larger than 2
					if len(pos_samples) < (C.num_rois // 2):
						selected_pos_samples = pos_samples.tolist()
					else:
						selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace = False).tolist()

					# Randomly choose negative samples (num_rois - num_pos)
					try:
						selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace = False).tolist()
					except:
						selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace = True).tolist()

					# Saving of all positive & negative samples in sel_samples
					sel_samples = selected_pos_samples + selected_neg_samples

				else:
					# For num_rois = 1 -> picking one positive or negative example
					selected_pos_samples = pos_samples.tolist()
					selected_neg_samples = neg_samples.tolist()

					if np.random.randint(0, 2):
						sel_samples = random.choice(neg_samples)
					else:
						sel_samples = random.choice(pos_samples)

				loss_class_v = model_classifier.test_on_batch([X_v, X2_v[:, sel_samples, :]], [Y1_v[:, sel_samples, :], Y2_v[:, sel_samples, :]])

				# Losses of training process
				losses2[iter_num_aux , 0] = loss_rpn_v[1]
				losses2[iter_num_aux, 1] = loss_rpn_v[2]

				losses2[iter_num_aux, 2] = loss_class_v[1]
				losses2[iter_num_aux, 3] = loss_class_v[2]
				losses2[iter_num_aux, 4] = loss_class_v[3]

				progbar.update(iter_num + 1, [('rpn_cls', np.mean(losses2[iter_num_aux, 0])), ('rpn_regr', np.mean(losses2[iter_num_aux, 1])), ('detector_cls', np.mean(losses2[iter_num_aux, 2])), ('detector_regr', np.mean(losses2[iter_num_aux, 3]))])
				
				iter_num += 1
				iter_num_aux += 1


				if iter_num == 1000 + len(val_imgs):
					loss_rpn_cls_v = np.mean(losses2[:, 0])
					loss_rpn_regr_v = np.mean(losses2[:, 1])
					loss_class_cls_v = np.mean(losses2[:, 2])
					loss_class_regr_v = np.mean(losses2[:, 3])
					class_acc_v = np.mean(losses2[:, 4])

					mean_overlapping_bboxes_v = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)					
					rpn_accuracy_for_epoch = []

					if C.verbose:
						print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes_v))
						print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc_v))
						print('Loss RPN classifier: {}'.format(loss_rpn_cls_v))
						print('Loss RPN regression: {}'.format(loss_rpn_regr_v))
						print('Loss Detector classifier: {}'.format(loss_class_cls_v))
						print('Loss Detector regression: {}'.format(loss_class_regr_v))
						print('Total loss: {}'.format(loss_rpn_cls_v + loss_rpn_regr_v + loss_class_cls_v + loss_class_regr_v))
						print('Elapsed time: {}'.format(time.time() - start_time))

						elapsed_time = (time.time() - start_time) / 60

					curr_loss_v = loss_rpn_cls_v + loss_rpn_regr_v + loss_class_cls_v + loss_class_regr_v



					start_time = time.time()

					new_row2 = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes_v, 3), 
                       			'class_acc':round(class_acc_v, 3), 
                    			'loss_rpn_cls':round(loss_rpn_cls_v, 3), 
                      			'loss_rpn_regr':round(loss_rpn_regr_v, 3), 
                       			'loss_class_cls':round(loss_class_cls_v, 3), 
                    			'loss_class_regr':round(loss_class_regr_v, 3), 
                    			'curr_loss':round(curr_loss_v, 3), 
                     			'elapsed_time':round(elapsed_time, 3), 
                       			'mAP': 0}

					#record2_df = record2_df.append(new_row2, ignore_index = True)
					#record2_df.to_csv(record_path2, index = 0)
					
					validate = False
					
					# Condition breaking loop
					i = False
					

		except Exception as e:
			print('Exception: {}'.format(e))
			continue

print('Training complete, exiting.')


