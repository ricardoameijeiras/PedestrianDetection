

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.backend import categorical_crossentropy

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

# Preventing divison by 0
epsilon = 1e-4


# Loss function -> regression RPN
def rpn_loss_regr(num_anchors):

	def rpn_loss_regr_fixed_num(y_true, y_pred):



		x = y_true[:, :, :, 4 * num_anchors:] - y_pred	      # Subtraction tesnors (true tensor & prediction tensor)!
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32) # true (x <= 1.0) | false (x > 1.0)

			# Loss function -> regression
		return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


# Loss function classification RPN -> Region proposals (positive & negative)
def rpn_loss_cls(num_anchors):

	def rpn_loss_cls_fixed_num(y_true, y_pred):

		# Loss function -> classification
		return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
		

	return rpn_loss_cls_fixed_num


# Loss function detection regression -> Region proposals & Ground truth boxes
def class_loss_regr(num_classes):

	def class_loss_regr_fixed_num(y_true, y_pred):

		x = y_true[:, :, 4 * num_classes:] - y_pred		      # Subtraction 2D Matrizes (true matrix & prediction matrix)!
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')	      # True (x <= 1.0) | False (x > 1.0)

		# Loss function -> regression
		return lambda_cls_regr * K.sum(y_true[:, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4 * num_classes])

	return class_loss_regr_fixed_num


# Loss function detection classification -> object classification	
def class_loss_cls(y_true, y_pred):

	# Loss function classification		
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))



