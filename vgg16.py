# -*- coding: utf-8 -*-


''' Python-Skript für das VGG16 Basis-Netzwerk (Subsampling Ratio r = 16)

    - Feature-Extrahierung wird auf den ersten 13 Convolutional-Layern ausgeführt
    - Aktivierungsfunktion als ReLu-Aktivierung (Rectified Linear Unit - Aktivierungsfunktion)
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
#from tensorflow.keras.engine.topology import get_source_inputs
#from tensorflow.keras.utils import layer_utils
#from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras import backend as K
from RoiPoolingConv import RoiPoolingConv

# Returning of path of pretrained weights -> VGG16
def get_weight_path():

	
	return '/home/benan/Detektion2/Model/vgg16_weights.h5'

# Return output-width & output-height of image (feature-map) after feature-extraction
def get_img_output_length(width, height):

	def get_output_length(input_length):
		return (input_length // 16)

	return get_output_length(width), get_output_length(height)    


def nn_base(input_tensor = None, trainable = False):

	input_shape = (None, None, 3)

	if input_tensor is None:
		img_input = Input(shape = input_shape)

	#else:
		#if not K.is_keras_tensor(input_tensor):
			#img_input = Input(tensor = input_tensor, shape = input_shape)
		
	else:
		img_input = input_tensor

	bn_axis = 3

	# VGG16 basis network -> 5 convolutional blocks
	# Convolutional Block 1
	x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1')(img_input)
	x = Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool')(x)

	# Convolutional Block 2
	x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1')(x)
	x = Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool')(x)

	# Convolutional Block 3
	x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool')(x)

	# Convolutional Block 4
	x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block4_pool')(x)

	# Convolutional Block 5
	x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation = 'relu', padding = 'same', name = 'block5_conv3')(x)
	# x = MaxPooling2D((2, 2), strides = (2, 2), name = 'block5_pool')(x)

	return x

# Visualization of Region-Proposal Networks
def rpn_layer(base_layers, num_anchors):

	x = Conv2D(512, (3, 3), padding = 'same', activation = 'relu', kernel_initializer = 'normal', name = 'rpn_conv1')(base_layers)

	# Convolution for class -> output (14 x 14 x 9)
	x_class = Conv2D(num_anchors, (1, 1), activation = 'sigmoid', kernel_initializer = 'uniform', name = 'rpn_out_class')(x)
	# Convolution for regression -> output (14 x 14 x 36)
	x_regr = Conv2D(num_anchors * 4, (1, 1), activation = 'linear', kernel_initializer = 'zero', name = 'rpn_out_regress')(x)

	return [x_class, x_regr, base_layers]

# Visualization of fully-connected network (classifier) -> Classification of ROI tensors & regression of Region-Proposals
def classifier_layer(base_layers, input_rois, num_rois, nb_classes = 3):

	input_shape = (num_rois, 7, 7, 512)
	pooling_regions = 7

	out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

	# Flattening of 3D ROI tensor (7 x 7 x 512) -> Input-Layer for fully-connected layer
	out = TimeDistributed(Flatten(name = 'flatten'))(out_roi_pool)

	# Fully-Connected layer 1 with Dropout
	out = TimeDistributed(Dense(4096, activation = 'relu', name = 'fc1'))(out)
	out = TimeDistributed(Dropout(0.5))(out)
	
	# Fully-Connected layer 2 with Dropout
	out = TimeDistributed(Dense(4096, activation = 'relu', name = 'fc2'))(out)
	out = TimeDistributed(Dropout(0.5))(out)

	# Classification layer with softmax activation
	out_class = TimeDistributed(Dense(nb_classes, activation = 'softmax', kernel_initializer = 'zero'), name = 'dense_class_{}'.format(nb_classes))(out)
	
	# Regression layer with linear activation -> Regression for background class not visualized (nb_classes - 1)
	out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation = 'linear', kernel_initializer = 'zero'), name = 'dense_regress_{}'.format(nb_classes))(out)

	return [out_class, out_regr]
