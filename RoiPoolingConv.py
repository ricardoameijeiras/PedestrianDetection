# -*- coding: utf-8 -*-

# Python-Skript für ROI Pooling -> Visualisierung als Schritt 2 nach RPN

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf

class RoiPoolingConv(Layer):

	'''
	# Argumente
	pool_size: int
		Ergebnis wird als 7x7-Output (pool_size = 7) visualisiert
	num_rois: int
	# Input-Größe
	Liste bestehend aus 4D Tensoren [X_img, X_roi] mit folgender Form:
	X_img:
	3D Tensor `(1, rows, cols, channels)` mit Form (rows x cols x channels) if dim_ordering = 'tf'.
	X_roi:
	ROI-Liste `(1,num_rois,4)` mit Form (num_rois x 4) -> 2D-Liste beinhaltet für jede ROI 4 Werte in Liste
	# Output-Größe
	4D Tensor `(1, num_rois, pool_size, pool_size, channels)` mit Form (num_rois x pool_size x pool_size x channels) if  	
	dim_ordering = 'tf'
	'''
	def __init__(self, pool_size, num_rois, **kwargs):

		#self.dim_ordering = K.set_image_dim_ordering('tf')

		'''
		# Konfigurations-Datei muss 'tf' beinhalten -> ansonsten wird Assertion-Error ausgeworfen
		assert self.dim_ordering in {'tf'}, 'dim_ordering must be in {tf}'''

		self.pool_size = pool_size
		self.num_rois = num_rois

		super(RoiPoolingConv, self).__init__(**kwargs)


	def build(self, input_shape):

		self.nb_channels = input_shape[0][3] # Parameter 'channels' aus Input-Größe [X_img, X_roi]


	def compute_output_shape(self, input_shape):

	# Tensorflow-Backend -> Channel-Zahl als letzte Dimension
		return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels


	def call(self, x, mask = None):

		# Assertion-Bedingung: Input-Größe als Liste/Array mit 2 Elementen (X_img, X_roi)!
		assert(len(x) == 2)

		img = x[0] # 3D Tensor -> (1, rows, cols, channels)
		rois = x[1] # 2D Tensor (Matrix) -> (1, num_rois, 4)

		input_shape = K.shape(img) # Shape (rows x cols x channels)

		# Initialisierung dynamischer Output-Liste für ROIs
		outputs = []

		# Für jedes Region-Proposal (ROI auf Feature-Map) wird Schleife durchlaufen -> Input-ROIs
		for roi_idx in range(self.num_rois):

			x = rois[0, roi_idx, 0] # Koordinate x für Ecke (oben links)
			y = rois[0, roi_idx, 1] # Koordinate y für Ecke (oben links)
			w = rois[0, roi_idx, 2] # Weite d. Region-Proposals
			h = rois[0, roi_idx, 3] # Höhe d. Region-Proposals

			
			x = K.cast(x, 'int32')
			y = K.cast(y, 'int32')
			w = K.cast(w, 'int32')
			h = K.cast(h, 'int32')

				# Erhöhung d. Effizienz 		
				# Slicing von ROI auf Feature-Map -> Anwendung einer Resize-Operation (pool_size x pool_size) statt Max-Pooling                                     
			rs = tf.image.resize_images(img[:, y:(y+h), x:(x+w), :], (self.pool_size, self.pool_size))
			outputs.append(rs) 

		# Output 3D Tensor-ROIs (7 x 7 x 512) werden als Einheit verkettet -> Achse 0
		final_output = K.concatenate(outputs, axis = 0)
		# 4D Tensor wird als (num_rois x pool_size x pool_size x nb_channels) visualisiert
		final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

		# Dimensions-Reihenfolge wird visualisiert -> Tensorflow-Backend
	# if self.dim_ordering == 'tf':
		final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

		return final_output
    
    
	def get_config(self):
		config = {'pool_size': self.pool_size,
		  	'num_rois': self.num_rois}
		base_config = super(RoiPoolingConv, self).get_config()

		return dict(list(base_config.items()) + list(config.items()))
