"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib import slim 

from utils import preprocessing

import numpy as np
import os
slim = tf.contrib.slim
L2_REG_SCALE = 5e-5

_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4


def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
	"""Atrous Spatial Pyramid Pooling.

	Args:
		inputs: A tensor of size [batch, height, width, channels].
		output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
			the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
		batch_norm_decay: The moving average decay when estimating layer activation
			statistics in batch normalization.
		is_training: A boolean denoting whether the input is for training.
		depth: The depth of the ResNet unit output.

	Returns:
		The atrous spatial pyramid pooling output.
	"""
	with tf.variable_scope("aspp"):
		if output_stride not in [8, 16]:
			raise ValueError('output_stride must be either 8 or 16.')

		atrous_rates = [6, 12, 18]
		if output_stride == 8:
			atrous_rates = [2 * rate for rate in atrous_rates]

		with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
			with arg_scope([layers.batch_norm], is_training=is_training):
				inputs_size = tf.shape(inputs)[1:3]
				# (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
				# the rates are doubled when output stride = 8.
				conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
				conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
				conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
				conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

				# (b) the image-level features
				with tf.variable_scope("image_level_features"):
					# global average pooling
					image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keep_dims=True)
					# 1x1 convolution with 256 filters( and batch normalization)
					image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
					# bilinearly upsample features
					image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

					net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
					net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

					return net


# def conv_relu(inputs,  depth, scope, rate=1, kernel=3, stride=1, ):

# 	with tf.variable_scope(scope):
# 		out = layers_lib.conv2d(inputs, depth, [kernel, kernel], rate=rate, stride=stride, scope='conv')
# 		out = tf.nn.relu(out)

# 	return out
def conv_relu(inputs,  depth, scope, rate=1, kernel=3, stride=1, is_separable=False):
	def split_separable_conv2d(inputs,
	                           filters,
	                           kernel_size=3,
	                           rate=1,
	                           weight_decay=0.00004,
	                           depthwise_weights_initializer_stddev=0.33,
	                           pointwise_weights_initializer_stddev=0.06,
	                           scope=None):
			outputs = slim.separable_conv2d(
	      	inputs,
	      	None,
	      	kernel_size=kernel_size,
	      	depth_multiplier=1,
	      	rate=rate,
	      	weights_initializer=tf.truncated_normal_initializer(stddev=depthwise_weights_initializer_stddev),
	      	weights_regularizer=None,
	      	scope=scope + '_depthwise')
			return slim.conv2d(outputs, filters, 1, weights_initializer=tf.truncated_normal_initializer(stddev=pointwise_weights_initializer_stddev), weights_regularizer=slim.l2_regularizer(weight_decay), scope=scope + '_pointwise')

	with tf.variable_scope(scope):
		# out = layers_lib.conv2d(inputs, depth, [kernel, kernel], rate=rate, stride=stride, )
		if is_separable == True:
			out = split_separable_conv2d(inputs, depth, kernel=kernel, rate=rate, scope='separable_conv')
		else:
			out = layers_lib.conv2d(inputs, depth, [kernel, kernel], rate=rate, stride=stride)
		out = tf.nn.relu(out)

	return out


def aspp_branch(sem_output, disp_output,  batch_norm_decay, is_training, is_separable=False, depth=20):

	with tf.variable_scope("aspp_branch"):

		atrous_rates = [6, 12, 18]

		with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
			with arg_scope([layers.batch_norm], is_training=is_training):
				inputs_size = tf.shape(sem_output)[1:3]

				# global 
				with tf.variable_scope("global_features"):
					global_sem_pre_conv = conv_relu(sem_output, depth , scope='sem_pre', kernel=3, stride=1)
					global_disp_pre_conv = conv_relu(disp_output, depth , scope='disp_pre', kernel=3, stride=1)
					global_sem_pool = tf.reduce_mean(global_sem_pre_conv, [1, 2], name='sem_pool', keep_dims=True)
					global_disp_pool = tf.reduce_mean(global_disp_pre_conv, [1, 2], name='disp_pool', keep_dims=True)

					global_pre_sum = global_sem_pool + global_disp_pool
					global_features = tf.image.resize_bilinear(global_pre_sum, inputs_size, name='upsample')

					global_features = conv_relu(global_features, depth, is_separable=is_separable,  scope='conv_1', kernel=3, stride=1)
					global_features = conv_relu(global_features, depth, is_separable=is_separable,  scope='conv_2', kernel=3, stride=1)
					global_features = conv_relu(global_features, depth, is_separable=is_separable,  scope='conv_3', kernel=3, stride=1)
					global_features = conv_relu(global_features, depth, is_separable=is_separable,  scope='conv_4', kernel=3, stride=1)

				# conv_1x1
				with tf.variable_scope("conv_1x1"):
					conv_1x1_sem_pre_conv = conv_relu(sem_output, depth , scope='sem_pre', kernel=1, stride=1)
					conv_1x1_disp_pre_conv = conv_relu(disp_output, depth , scope='disp_pre', kernel=1, stride=1)
					
					conv_1x1_features = conv_1x1_sem_pre_conv + conv_1x1_disp_pre_conv
					
					conv_1x1_features = conv_relu(conv_1x1_features, depth, is_separable=is_separable,  scope='conv_1', kernel=3, stride=1)
					conv_1x1_features = conv_relu(conv_1x1_features, depth, is_separable=is_separable,  scope='conv_2', kernel=3, stride=1)
					conv_1x1_features = conv_relu(conv_1x1_features, depth, is_separable=is_separable,  scope='conv_3', kernel=3, stride=1)
					conv_1x1_features = conv_relu(conv_1x1_features, depth, is_separable=is_separable,  scope='conv_4', kernel=3, stride=1)

				# conv_3x3_rate_1
				with tf.variable_scope("conv_3x3_rate_1"):
					conv_3x3_rate_1_sem_pre_conv = conv_relu(sem_output, depth, is_separable=is_separable, scope='sem_pre', rate=atrous_rates[0],  kernel=3, stride=1)
					conv_3x3_rate_1_disp_pre_conv = conv_relu(disp_output, depth, is_separable=is_separable,  scope='disp_pre', rate=atrous_rates[0], kernel=3, stride=1)
					
					conv_3x3_rate_1_features = conv_3x3_rate_1_sem_pre_conv + conv_3x3_rate_1_disp_pre_conv

					conv_3x3_rate_1_features = conv_relu(conv_3x3_rate_1_features, depth, is_separable=is_separable,  scope='conv_1', kernel=3, stride=1)
					conv_3x3_rate_1_features = conv_relu(conv_3x3_rate_1_features, depth, is_separable=is_separable,  scope='conv_2', kernel=3, stride=1)
					conv_3x3_rate_1_features = conv_relu(conv_3x3_rate_1_features, depth, is_separable=is_separable,  scope='conv_3', kernel=3, stride=1)
					conv_3x3_rate_1_features = conv_relu(conv_3x3_rate_1_features, depth, is_separable=is_separable,  scope='conv_4', kernel=3, stride=1)

				# conv_3x3_rate_2
				with tf.variable_scope("conv_3x3_rate_2"):
					conv_3x3_rate_2_sem_pre_conv = conv_relu(sem_output, depth, is_separable=is_separable, rate=atrous_rates[1], scope='sem_pre', kernel=3, stride=1)
					conv_3x3_rate_2_disp_pre_conv = conv_relu(disp_output, depth, is_separable=is_separable, rate=atrous_rates[1], scope='disp_pre', kernel=3, stride=1)
					conv_3x3_rate_2_features = conv_3x3_rate_2_sem_pre_conv + conv_3x3_rate_2_disp_pre_conv

					conv_3x3_rate_2_features = conv_relu(conv_3x3_rate_2_features, depth, is_separable=is_separable,  scope='conv_1', kernel=3, stride=1)
					conv_3x3_rate_2_features = conv_relu(conv_3x3_rate_2_features, depth, is_separable=is_separable,  scope='conv_2', kernel=3, stride=1)
					conv_3x3_rate_2_features = conv_relu(conv_3x3_rate_2_features, depth, is_separable=is_separable,  scope='conv_3', kernel=3, stride=1)
					conv_3x3_rate_2_features = conv_relu(conv_3x3_rate_2_features, depth, is_separable=is_separable,  scope='conv_4', kernel=3, stride=1)

				# conv_3x3_rate_3
				with tf.variable_scope("conv_3x3_rate_3"):
					conv_3x3_rate_3_sem_pre_conv = conv_relu(sem_output, depth, is_separable=is_separable, rate=atrous_rates[2], scope='sem_pre', kernel=3, stride=1)
					conv_3x3_rate_3_disp_pre_conv = conv_relu(disp_output, depth, is_separable=is_separable, rate=atrous_rates[2], scope='disp_pre', kernel=3, stride=1)
					
					conv_3x3_rate_3_features = conv_3x3_rate_3_sem_pre_conv + conv_3x3_rate_3_disp_pre_conv

					conv_3x3_rate_3_features = conv_relu(conv_3x3_rate_3_features, depth, is_separable=is_separable,  scope='conv_1', kernel=3, stride=1)
					conv_3x3_rate_3_features = conv_relu(conv_3x3_rate_3_features, depth, is_separable=is_separable,  scope='conv_2', kernel=3, stride=1)
					conv_3x3_rate_3_features = conv_relu(conv_3x3_rate_3_features, depth, is_separable=is_separable,  scope='conv_3', kernel=3, stride=1)
					conv_3x3_rate_3_features = conv_relu(conv_3x3_rate_3_features, depth, is_separable=is_separable,  scope='conv_4', kernel=3, stride=1)

				net = tf.concat([global_features, conv_1x1_features, conv_3x3_rate_1_features, conv_3x3_rate_2_features, conv_3x3_rate_3_features], axis=3, name='concat')
				net = conv_relu(net, depth*5, scope='sum_conv', stride=1, kernel=3, rate=1)

				return net



def deeplab_v3_plus_generator(num_classes,
	                              output_stride,
	                              base_architecture,
	                              pre_trained_model,
	                              aux_ratio,
	                              batch_norm_decay=_BATCH_NORM_DECAY,
	                              data_format='channels_last'):
	"""Generator for DeepLab v3 plus models.

	Args:
		num_classes: The number of possible classes for image classification.
		output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
			the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
		base_architecture: The architecture of base Resnet building block.
		pre_trained_model: The path to the directory that contains pre-trained models.
		batch_norm_decay: The moving average decay when estimating layer activation
			statistics in batch normalization.
		data_format: The input format ('channels_last', 'channels_first', or None).
			If set to None, the format is dependent on whether a GPU is available.
			Only 'channels_last' is supported currently.

	Returns:
		The model function that takes in `inputs` and `is_training` and
		returns the output tensor of the DeepLab v3 model.
	"""

	if data_format is None:
		# data_format = (
		#     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
		pass

	if batch_norm_decay is None:
		batch_norm_decay = _BATCH_NORM_DECAY

	if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
		raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_50'.")

	if base_architecture == 'resnet_v2_50':
		base_model = resnet_v2.resnet_v2_50
	else:
		base_model = resnet_v2.resnet_v2_101

	def model(inputs, is_training):
		"""Constructs the ResNet model given the inputs."""
		if data_format == 'channels_first':
			# Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
			# This provides a large performance boost on GPU. See
			# https://www.tensorflow.org/performance/performance_guide#data_formats
			inputs = tf.transpose(inputs, [0, 3, 1, 2])

		# tf.logging.info('net shape: {}'.format(inputs.shape))
		# encoder
		with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
			logits, end_points = base_model(inputs,
			                                num_classes=None,
			                                is_training=is_training,
			                                global_pool=False,
			                                output_stride=output_stride)


		inputs_size = tf.shape(inputs)[1:3]
		net = end_points[base_architecture + '/block4']

		aspp = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)

		with tf.variable_scope("decoder"):
			with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
				with arg_scope([layers.batch_norm], is_training=is_training):
					with tf.variable_scope("low_level_features"):
						low_level_features = end_points[base_architecture + '/block1/unit_3/bottleneck_v2/conv1']
						low_level_features = layers_lib.conv2d(low_level_features, 48,
						                                       [1, 1], stride=1, scope='conv_1x1')
						low_level_features_size = tf.shape(low_level_features)[1:3]

						net_4x = tf.image.resize_bilinear(aspp, low_level_features_size, name='upsample_1')
						net_cat = tf.concat([net_4x, low_level_features], axis=3, name='concat')

					with tf.variable_scope("sem_upsampling_logits"):
						sem_net = layers_lib.conv2d(net_cat, 256, [3, 3], stride=1, scope='conv_3x3_1')
						sem_net = layers_lib.conv2d(sem_net, 256, [3, 3], stride=1, scope='conv_3x3_2')
						aux_sem_net = layers_lib.conv2d(sem_net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
						aux_sem_net = tf.image.resize_bilinear(aux_sem_net, inputs_size, name='upsample_2')

					with tf.variable_scope("disp_upsampling_logits"):

						disp_net = layers_lib.conv2d(net_cat, 256, [3,3], stride=1)
						disp_net = layers_lib.conv2d(disp_net, 256, [3,3], stride=1)
						aux_disp_net = layers_lib.conv2d(disp_net, 1, (1, 1), padding='SAME')
						aux_disp_net = tf.image.resize_bilinear(aux_disp_net, inputs_size, name='upsample_2')


					sum_conv = aspp_branch(sem_net, disp_net ,batch_norm_decay=_BATCH_NORM_DECAY, is_training=is_training, depth=20)

					sem_net = layers_lib.conv2d(sum_conv, num_classes, [3, 3], stride=1, scope='sem_conv')
					sem_net = tf.image.resize_bilinear(sem_net, inputs_size, name='sem_upsample')

					disp_net = layers_lib.conv2d(sum_conv, 1, [3, 3], stride=1, scope='disp_conv')
					disp_net = tf.image.resize_bilinear(disp_net, inputs_size, name='disp_upsample')	

			if is_training and not os.path.exists('./model/cs/0.85/' + str(aux_ratio)):
				exclude = [base_architecture + '/logits', 'global_step', 'aspp_branch']
				variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
				tf.train.init_from_checkpoint(pre_trained_model,
				                              {v.name.split(':')[0]: v for v in variables_to_restore})

			return aux_sem_net, aux_disp_net, sem_net, disp_net


		# with tf.variable_scope("decoder"):
		# 	with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
		# 		with arg_scope([layers.batch_norm], is_training=is_training):
		# 			with tf.variable_scope("low_level_features"):
		# 				low_level_features = end_points[base_architecture + '/block1/unit_3/bottleneck_v2/conv1']
		# 				low_level_features = layers_lib.conv2d(low_level_features, 48,
		# 				                                       [1, 1], stride=1, scope='conv_1x1')
		# 				low_level_features_size = tf.shape(low_level_features)[1:3]

		# 			# with tf.variable_scope("sem_upsample"):
		# 			# 	# sem_aspp = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)
		# 			# 	# sem_net_4x = tf.image.resize_bilinear(sem_aspp, low_level_features_size, name='upsample_1')
		# 			# 	sem_net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='1x_conv1')
		# 			# 	sem_net = layers_lib.conv2d(sem_net, 256, [3, 3], stride=1, scope='1x_conv2')

		# 			# 	sem_net_4x = tf.image.resize_bilinear(sem_net, low_level_features_size, name='upsample1')

		# 			# 	sem_net_4x = layers_lib.conv2d(sem_net_4x, 256, [3, 3], stride=1, scope='4x_conv1')
		# 			# 	sem_net_4x = layers_lib.conv2d(sem_net_4x, num_classes, [3, 3], stride=1, scope='4x_conv2')

		# 			# 	sem_net_16x = tf.image.resize_bilinear(sem_net_4x, inputs_size, name='16x_upsample2')


		# 			# with tf.variable_scope("disp_upsample"):
		# 			# 	# sem_aspp = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)
		# 			# 	# sem_net_4x = tf.image.resize_bilinear(sem_aspp, low_level_features_size, name='upsample_1')
		# 			# 	disp_net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='1x_conv1')
		# 			# 	disp_net = layers_lib.conv2d(disp_net, 256, [3, 3], stride=1, scope='1x_conv2')

		# 			# 	disp_net_4x = tf.image.resize_bilinear(disp_net, low_level_features_size, name='upsample1')

		# 			# 	disp_net_4x = layers_lib.conv2d(disp_net_4x, 256, [3, 3], stride=1, scope='4x_conv1')
		# 			# 	disp_net_4x = layers_lib.conv2d(disp_net_4x, 1, [3, 3], stride=1, scope='4x_conv2')

		# 			# 	disp_net_16x = tf.image.resize_bilinear(disp_net_4x, inputs_size, name='16x_upsample2')

		# 			with tf.variable_scope("sem_upsample"):
		# 				# sem_aspp = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)
		# 				# sem_net_4x = tf.image.resize_bilinear(sem_aspp, low_level_features_size, name='upsample_1')
		# 				sem_net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='1x_conv1')
		# 				sem_net = layers_lib.conv2d(sem_net, 256, [3, 3], stride=1, scope='1x_conv2')

		# 				sem_net_4x = tf.image.resize_bilinear(sem_net, low_level_features_size, name='upsample1')

		# 				sem_net_4x = layers_lib.conv2d(sem_net_4x, 256, [3, 3], stride=1, scope='4x_conv1')
		# 				aux_sem_net_4x = layers_lib.conv2d(sem_net_4x, num_classes, [3, 3], stride=1, scope='4x_conv2')

		# 				aux_sem_net_16x = tf.image.resize_bilinear(aux_sem_net_4x, inputs_size, name='16x_upsample2')


		# 			with tf.variable_scope("disp_upsample"):
		# 				# sem_aspp = atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)
		# 				# sem_net_4x = tf.image.resize_bilinear(sem_aspp, low_level_features_size, name='upsample_1')
		# 				disp_net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='1x_conv1')
		# 				disp_net = layers_lib.conv2d(disp_net, 256, [3, 3], stride=1, scope='1x_conv2')

		# 				disp_net_4x = tf.image.resize_bilinear(disp_net, low_level_features_size, name='upsample1')

		# 				disp_net_4x = layers_lib.conv2d(disp_net_4x, 256, [3, 3], stride=1, scope='4x_conv1')
		# 				aux_disp_net_4x = layers_lib.conv2d(disp_net_4x, 1, [3, 3], stride=1, scope='4x_conv2')

		# 				aux_disp_net_16x = tf.image.resize_bilinear(aux_disp_net_4x, inputs_size, name='16x_upsample2')

		# 			sum_conv = aspp_branch(sem_net_4x, disp_net_4x ,batch_norm_decay=_BATCH_NORM_DECAY, is_training=is_training, depth=20)

		# 			sem_net_4x = layers_lib.conv2d(sum_conv, num_classes, [3, 3], stride=1, scope='sem_conv')
		# 			sem_net_16x = tf.image.resize_bilinear(sem_net_4x, inputs_size, name='sem_upsample')

		# 			disp_net_4x = layers_lib.conv2d(sum_conv, 1, [3, 3], stride=1, scope='disp_conv')
		# 			disp_net_16x = tf.image.resize_bilinear(disp_net_4x, inputs_size, name='disp_upsample')				





	return model


def deeplabv3_plus_model_fn(features, labels, mode, params):
	"""Model function for PASCAL VOC."""
	if isinstance(features, dict):
		features = features['feature']

	images = tf.cast(
		tf.map_fn(preprocessing.mean_image_addition, features),
		tf.uint8)

	network = deeplab_v3_plus_generator(params['num_classes'],
	                                    params['output_stride'],
	                                    params['base_architecture'],
	                                    params['pre_trained_model'],
	                                    params['aux_ratio'],
	                                    params['batch_norm_decay'])

	aux_sem_logits, aux_disp_logits, sem_logits, disp_logits = network(features, mode == tf.estimator.ModeKeys.TRAIN)

	sem_pred_classes = tf.expand_dims(tf.argmax(sem_logits, axis=3, output_type=tf.int32), axis=3)
	aux_sem_pred_classes = tf.expand_dims(tf.argmax(aux_sem_logits, axis=3, output_type=tf.int32), axis=3)

	sem_pred_decoded = tf.py_func(preprocessing.decode_sem, [sem_pred_classes, params['batch_size'], params['num_classes']], tf.uint8)
	aux_sem_pred_decoded = tf.py_func(preprocessing.decode_sem, [aux_sem_pred_classes, params['batch_size'], params['num_classes']], tf.uint8)
	
	disp_pred_decoded = tf.cast(preprocessing.decode_disp(disp_logits, params['dataset']), tf.uint8)
	aux_disp_pred_decoded = tf.cast(preprocessing.decode_disp(aux_disp_logits, params['dataset']), tf.uint8)

	predictions = {
		'image': images,
		'classes': sem_pred_classes,
		'aux_classes': aux_sem_pred_classes,
		'probabilities': tf.nn.softmax(sem_logits, name='softmax_tensor'),
		'sem_decoded': sem_pred_decoded,
		'disp_decoded': disp_pred_decoded,
		'aux_sem_decoded': aux_sem_pred_decoded,
		'aux_disp_decoded': aux_disp_pred_decoded
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		# Delete 'decoded_labels' from predictions because custom functions produce error when used with saved_model
		predictions_without_decoded_labels = predictions.copy()
		del predictions_without_decoded_labels['sem_decoded']
		del predictions_without_decoded_labels['disp_decoded']


		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=predictions,
			export_outputs={
				'preds': tf.estimator.export.PredictOutput(
					predictions_without_decoded_labels)
			})

	gt_decoded_sem = tf.py_func(preprocessing.decode_sem,
	                               [labels['sem'], params['batch_size'], params['num_classes']], tf.uint8)
	gt_decoded_disp = tf.cast(preprocessing.decode_disp(labels['disp'], params['dataset']), tf.uint8)

	sem_labels = tf.squeeze(labels['sem'], axis=3)  # reduce the channel dimension.

	sem_logits_by_num_classes = tf.reshape(sem_logits, [-1, params['num_classes']])
	aux_sem_logits_by_num_classes = tf.reshape(aux_sem_logits, [-1, params['num_classes']])
	sem_labels_flat = tf.reshape(sem_labels, [-1, ])

	sem_valid_indices = tf.to_int32(sem_labels_flat <= params['num_classes'] - 1)
	sem_valid_logits = tf.dynamic_partition(sem_logits_by_num_classes, sem_valid_indices, num_partitions=2)[1]
	aux_sem_valid_logits = tf.dynamic_partition(aux_sem_logits_by_num_classes, sem_valid_indices, num_partitions=2)[1]
	sem_valid_labels = tf.dynamic_partition(sem_labels_flat, sem_valid_indices, num_partitions=2)[1]

	sem_preds_flat = tf.reshape(sem_pred_classes, [-1, ])
	aux_sem_preds_flat = tf.reshape(aux_sem_pred_classes, [-1, ])
	sem_valid_preds = tf.dynamic_partition(sem_preds_flat, sem_valid_indices, num_partitions=2)[1]
	aux_sem_valid_preds = tf.dynamic_partition(aux_sem_preds_flat, sem_valid_indices, num_partitions=2)[1]
	confusion_matrix = tf.confusion_matrix(sem_valid_labels, sem_valid_preds, num_classes=params['num_classes'])

	predictions['sem_valid_preds'] = sem_valid_preds
	predictions['sem_valid_labels'] = sem_valid_labels
	predictions['confusion_matrix'] = confusion_matrix

	sem_loss = tf.losses.sparse_softmax_cross_entropy(
		logits=sem_valid_logits, labels=sem_valid_labels)

	aux_sem_loss = tf.losses.sparse_softmax_cross_entropy(
		logits=aux_sem_valid_logits, labels=sem_valid_labels)

	# Create a tensor named cross_entropy for logging purposes.
	tf.identity(sem_loss, name='sem_loss')
	tf.summary.scalar('sem_loss', sem_loss)

	tf.identity(aux_sem_loss, name='aux_sem_loss')
	tf.summary.scalar('aux_sem_loss', aux_sem_loss)

	def berhu_loss(x, y):
		x_flat = tf.reshape(x, [-1, ])
		y_flat = tf.reshape(y, [-1, ])

		valid_indices = tf.to_int32(x_flat >= 0)

		valid_x = tf.dynamic_partition(x_flat, valid_indices, num_partitions=2)[1]

		valid_y = tf.dynamic_partition(y_flat, valid_indices, num_partitions=2)[1]

		# valid_x = tf.Print(valid_x, [valid_x], 'valid_x', summarize=200)
		# valid_y = tf.Print(valid_y, [valid_y], 'valid_y', summarize=200)

		abs_error = tf.abs(valid_x - valid_y)
		c = 0.2 * tf.reduce_max(abs_error)

		berhuloss = tf.reduce_mean(tf.where(abs_error <= c, abs_error, (tf.square(abs_error) + tf.square(c)) / (2 * c)))

		return berhuloss

	def SILog_loss(x, y):
		x_flat = tf.reshape(x, [-1, ])
		y_flat = tf.reshape(y, [-1, ])

		valid_indices = tf.to_int32(x_flat >= 0)

		valid_x = tf.dynamic_partition(x_flat, valid_indices, num_partitions=2)[1]

		valid_y = tf.dynamic_partition(y_flat, valid_indices, num_partitions=2)[1]


		valid_x = preprocessing.unnormalization(valid_x, params['dataset'])
		valid_y = preprocessing.unnormalization(valid_y, params['dataset'])


		log_1 = tf.log(valid_y + 1.)
		log_2 = tf.log(valid_x + 1.)
		silog_loss = tf.reduce_mean(tf.square(log_1 - log_2), axis=-1) - 1/2 * tf.square(tf.reduce_mean(log_1 - log_2, axis=-1))

		return silog_loss

	def threshold_self(pred, label, diff=1.25):

		label_by_pred = pred / label
		max_div = tf.where(label_by_pred > 1, label_by_pred, 1 / label_by_pred)

		good_result_count = tf.reduce_sum(tf.to_int32(max_div < diff))
		all_result_count = tf.shape(max_div)[0]

		return tf.to_float(good_result_count / all_result_count)


	# def mean_error_self(pred, label):
	#
	#
	#
	#
	# 	abs_error = tf.reduce_mean(tf.abs(valid_label - valid_pred))
	#
	# 	return tf.to_float(abs_error)


	# disp_berhu_loss = berhu_loss(labels['disp'], disp_logits)
	disp_SILog_loss = SILog_loss(labels['disp'], disp_logits)
	aux_disp_SILog_loss = SILog_loss(labels['disp'], aux_disp_logits)
	# disp_thresold_self = threshold_self(disp_logits, labels['disp'], diff=5)
	# disp_mean_error_self = mean_error_self(disp_logits, labels['disp'])

	# tf.identity(disp_berhu_loss, name='disp_berhu_loss')
	# tf.summary.scalar('disp_berhu_loss', disp_berhu_loss)

	tf.identity(disp_SILog_loss, name='disp_SILog_loss')
	tf.summary.scalar('disp_SILog_loss', disp_SILog_loss)

	tf.identity(aux_disp_SILog_loss, name='aux_disp_SILog_loss')
	tf.summary.scalar('aux_disp_SILog_loss', aux_disp_SILog_loss)

	if not params['freeze_batch_norm']:
		train_var_list = [v for v in tf.trainable_variables()]
	else:
		train_var_list = [v for v in tf.trainable_variables()
		                  if 'beta' not in v.name and 'gamma' not in v.name]

	if params['ratio'] == 0:
		train_var_list = [v for v in train_var_list if 'sem_upsampling_logits' not in v.name]
	elif params['ratio'] == 1:
		train_var_list = [v for v in train_var_list if 'disp_upsampling_logits' not in v.name]

	if params['aux_ratio'] == 1:
		train_var_list = [v for v in train_var_list if 'aspp_branch' not in v.name]
	# elif params['aux_ratio'] == 0:
	# 	train_var_list = [v for v in train_var_list if 'aspp_branch' in v.name]


	# Add weight decay to the loss.
	with tf.variable_scope("total_loss"):

		aux_ratio = params['aux_ratio']
		loss = params['ratio'] * ((1 - aux_ratio) * sem_loss + aux_ratio * aux_sem_loss )+ (1-params['ratio']) * ((1 - aux_ratio) * disp_SILog_loss + aux_ratio * aux_disp_SILog_loss) + params.get('weight_decay', _WEIGHT_DECAY) * tf.add_n(
			[tf.nn.l2_loss(v) for v in train_var_list])
	# loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

	if mode == tf.estimator.ModeKeys.TRAIN:
		tf.summary.image('images',
		                 tf.concat(axis=2, values=[images, gt_decoded_sem, sem_pred_decoded, aux_sem_pred_decoded, preprocessing.disp_to_show(gt_decoded_disp), preprocessing.disp_to_show(disp_pred_decoded), preprocessing.disp_to_show(aux_disp_pred_decoded)]),
		                 max_outputs=params['tensorboard_images_max_outputs'])  # Concatenate row-wise.

		global_step = tf.train.get_or_create_global_step()

		if params['learning_rate_policy'] == 'piecewise':
			# Scale the learning rate linearly with the batch size. When the batch size
			# is 128, the learning rate should be 0.1.
			initial_learning_rate = 0.1 * params['batch_size'] / 128
			batches_per_epoch = params['num_train'] / params['batch_size']
			# Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
			boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
			values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
			learning_rate = tf.train.piecewise_constant(
				tf.cast(global_step, tf.int32), boundaries, values)
		elif params['learning_rate_policy'] == 'poly':
			learning_rate = tf.train.polynomial_decay(
				params['initial_learning_rate'],
				tf.cast(global_step, tf.int32) - params['initial_global_step'],
				params['max_iter'], params['end_learning_rate'], power=params['power'])
		else:
			raise ValueError('Learning rate policy must be "piecewise" or "poly"')

		# Create a tensor named learning_rate for logging purposes
		tf.identity(learning_rate, name='learning_rate')
		tf.summary.scalar('learning_rate', learning_rate)

		optimizer = tf.train.MomentumOptimizer(
			learning_rate=learning_rate,
			momentum=params['momentum'])

		# Batch norm requires update ops to be added as a dependency to the train_op
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)
	else:
		train_op = None

	sem_accuracy = tf.metrics.accuracy(
		sem_valid_labels, sem_valid_preds)
	sem_mean_iou = tf.metrics.mean_iou(sem_valid_labels, sem_valid_preds, params['num_classes'])
	aux_sem_accuracy = tf.metrics.accuracy(
		sem_valid_labels, aux_sem_valid_preds)
	aux_sem_mean_iou = tf.metrics.mean_iou(sem_valid_labels, aux_sem_valid_preds, params['num_classes'])

	x_flat = tf.reshape(labels['disp'], [-1, ])
	y_flat = tf.reshape(disp_logits, [-1, ])
	aux_y_flat = tf.reshape(aux_disp_logits, [-1, ])

	valid_indices = tf.to_int32(x_flat >= 0)

	valid_x = tf.dynamic_partition(x_flat, valid_indices, num_partitions=2)[1]

	valid_y = tf.dynamic_partition(y_flat, valid_indices, num_partitions=2)[1]
	aux_valid_y = tf.dynamic_partition(aux_y_flat, valid_indices, num_partitions=2)[1]

	un_disp_label = preprocessing.unnormalization(valid_x, params['dataset'])
	un_disp_pred = preprocessing.unnormalization(valid_y, params['dataset'])
	aux_un_disp_pred = preprocessing.unnormalization(valid_y, params['dataset'])
	# disp_threshold_sub_5 = tf.metrics.percentage_below(tf.abs(un_disp_label - un_disp_pred), threshold=5)
	# disp_threshold_sub_3 = tf.metrics.percentage_below(tf.abs(un_disp_label - un_disp_pred), threshold=3)
	# disp_threshold_sub_1 = tf.metrics.percentage_below(tf.abs(un_disp_label - un_disp_pred), threshold=1)
	disp_rse_linear = tf.metrics.root_mean_squared_error(un_disp_label, un_disp_pred)
	aux_disp_rse_linear = tf.metrics.root_mean_squared_error(un_disp_label, aux_un_disp_pred)
	disp_rse_log = tf.metrics.root_mean_squared_error(tf.log(un_disp_label), tf.log(un_disp_pred))
	aux_disp_rse_log = tf.metrics.root_mean_squared_error(tf.log(un_disp_label), tf.log(aux_un_disp_pred))
	disp_mse = tf.metrics.mean_absolute_error(un_disp_label, un_disp_pred)
	aux_disp_mse = tf.metrics.mean_absolute_error(un_disp_label, aux_un_disp_pred)


	#
	# x_ori_flat = tf.reshape(labels['disp_ori'], [-1, ])
	# disp_logits_ori = tf.image.resize_bilinear(disp_logits, tf.shape(labels['disp_ori'])[1:3])
	# y_ori_flat = tf.reshape(preprocessing.unnormalization(disp_logits_ori, params['dataset']), [-1, ])
	#
	# valid_ori_indices = tf.to_int32(x_ori_flat > 0)
	# valid_ori_x = tf.dynamic_partition(x_ori_flat, valid_ori_indices, num_partitions=2)[1]
	# valid_ori_y = tf.dynamic_partition(y_ori_flat, valid_ori_indices, num_partitions=2)[1]
	# disp_rse_linear_ori = tf.metrics.root_mean_squared_error(valid_ori_y, valid_ori_x)
	# disp_mse_ori = tf.metrics.mean_absolute_error(valid_ori_y, valid_ori_x)

	def log10(x):
		numerator = tf.log(x)
		denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
		return numerator / denominator

	# predictions['abs_ori'] = tf.reduce_mean(tf.abs(valid_ori_y - valid_ori_x))
	# predictions['rms_ori'] = tf.sqrt(tf.reduce_mean(tf.square(valid_ori_y - valid_ori_x)))
	# predictions['rel_abs_ori'] = tf.reduce_mean(tf.abs(valid_ori_y - valid_ori_x)/valid_ori_y)
	# predictions['rel_sqr_ori'] = tf.sqrt(tf.reduce_mean(tf.square(valid_ori_y - valid_ori_x)/ valid_ori_y))
	# predictions['log10_ori'] = tf.reduce_mean(tf.sqrt(tf.square(log10(valid_ori_y) - log10(valid_ori_x))))
	# predictions['e1_ori'] = threshold_self(valid_ori_y, valid_ori_x, diff=1.25)
	# predictions['e2_ori'] = threshold_self(valid_ori_y, valid_ori_x, diff=1.25 * 1.25)
	# predictions['e3_ori'] = threshold_self(valid_ori_y, valid_ori_x, diff=1.25 * 1.25 * 1.25)
	#
	# predictions['abs'] = tf.reduce_mean(tf.abs(valid_y - valid_x))
	# predictions['rms'] = tf.sqrt(tf.reduce_mean(tf.square(valid_y - valid_x)))
	# predictions['rel_abs'] = tf.reduce_mean(tf.abs(valid_y - valid_x)/valid_y)
	# predictions['rel_sqr'] = tf.sqrt(tf.reduce_mean(tf.square(valid_y - valid_x)/ valid_y))
	# predictions['log10'] = tf.reduce_mean(tf.sqrt(tf.square(log10(valid_y) - log10(valid_x))))
	# predictions['e1'] = threshold_self(valid_y, valid_x, diff=1.25)
	# predictions['e2'] = threshold_self(valid_y, valid_x, diff=1.25 * 1.25)
	# predictions['e3'] = threshold_self(valid_y, valid_x, diff=1.25 * 1.25 * 1.25)

	# compute threshold div11
	label_by_pred = un_disp_label / un_disp_pred
	max_div = tf.where(label_by_pred > 1, label_by_pred, 1/label_by_pred)

	# disp_threshold_div_e3 = tf.metrics.percentage_below(max_div, threshold=1.25*1.25*1.25)
	# disp_threshold_div_e2 = tf.metrics.percentage_below(max_div, threshold=1.25*1.25)
	# disp_threshold_div_e1 = tf.metrics.percentage_below(max_div, threshold=1.25)

	# compute relative difference
	# abs_relative_diff = tf.metrics.mean(tf.abs(un_disp_pred - un_disp_label)/un_disp_label)
	# squared_relative_diff = tf.metrics.mean(tf.square(un_disp_pred - un_disp_label)/un_disp_label)

	metrics = {
		'sem_px_accuracy': sem_accuracy,
		'sem_mean_iou': sem_mean_iou,
		'aux_sem_px_accuracy': aux_sem_accuracy,
		'aux_sem_mean_iou': aux_sem_mean_iou,
		# 'disp_threshold_5': disp_threshold_sub_5,
		# 'disp_threshold_3': disp_threshold_sub_3,
		# 'disp_threshold_1': disp_threshold_sub_1,
		# 'disp_threshold_div_e3': disp_threshold_div_e3,
		# 'disp_threshold_div_e2': disp_threshold_div_e2,
		# 'disp_threshold_div_e1': disp_threshold_div_e1,
		'disp_rse_linear': disp_rse_linear,
		'disp_rse_log': disp_rse_log,
		'aux_disp_rse_linear': aux_disp_rse_linear,
		'aux_disp_rse_log': aux_disp_rse_log,

		# 'abs_relative_diff': abs_relative_diff,
		# 'squared_relative_diff': squared_relative_diff,
		# 'disp_mse': disp_mse

	}

	# Create a tensor named train_accuracy for logging purposes
	tf.identity(sem_accuracy[1], name='sem_train_px_accuracy')
	tf.summary.scalar('sem_train_px_accuracy', sem_accuracy[1])

	tf.identity(aux_sem_accuracy[1], name='aux_sem_train_px_accuracy')
	tf.summary.scalar('aux_sem_train_px_accuracy', aux_sem_accuracy[1])

	# tf.identity(disp_threshold_sub_1[1], name='disp_threshold_sub_1')
	# tf.summary.scalar('disp_threshold_sub_1', disp_threshold_sub_1[1])

	# tf.identity(disp_threshold_sub_3[1], name='disp_threshold_sub_3')
	# tf.summary.scalar('disp_threshold_sub_3', disp_threshold_sub_3[1])

	# tf.identity(disp_threshold_sub_5[1], name='disp_threshold_sub_5')
	# tf.summary.scalar('disp_threshold_sub_5', disp_threshold_sub_5[1])

	tf.identity(disp_rse_linear[1], name='disp_rse_linear')
	tf.summary.scalar('disp_rse_linear', disp_rse_linear[1])

	tf.identity(disp_rse_log[1], name='disp_rse_log')
	tf.summary.scalar('disp_rse_log', disp_rse_log[1])

	tf.identity(aux_disp_rse_linear[1], name='aux_disp_rse_linear')
	tf.summary.scalar('aux_disp_rse_linear', aux_disp_rse_linear[1])

	tf.identity(aux_disp_rse_log[1], name='aux_disp_rse_log')
	tf.summary.scalar('aux_disp_rse_log', aux_disp_rse_log[1])

	# tf.identity(disp_mse[1], name='disp_mse')
	# tf.summary.scalar('disp_mse', disp_mse[1])

	# tf.identity(abs_relative_diff[1], name='abs_relative_diff')
	# tf.summary.scalar('abs_relative_diff', abs_relative_diff[1])

	# tf.identity(squared_relative_diff[1], name='squared_relative_diff')
	# tf.summary.scalar('squared_relative_diff', squared_relative_diff[1])

	# tf.identity(disp_threshold_div_e3[1], name='disp_threshold_div_e3')
	# tf.summary.scalar('disp_threshold_div_e3', disp_threshold_div_e3[1])

	# tf.identity(disp_threshold_div_e2[1], name='disp_threshold_div_e2')
	# tf.summary.scalar('disp_threshold_div_e2', disp_threshold_div_e2[1])

	# tf.identity(disp_threshold_div_e1[1], name='disp_threshold_div_e1')
	# tf.summary.scalar('disp_threshold_div_e1', disp_threshold_div_e1[1])


	def compute_mean_iou(total_cm, name='sem_mean_iou'):
		"""Compute the mean intersection-over-union via the confusion matrix."""
		sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
		sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
		cm_diag = tf.to_float(tf.diag_part(total_cm))
		denominator = sum_over_row + sum_over_col - cm_diag

		# The mean is only computed over classes that appear in the
		# label or prediction tensor. If the denominator is 0, we need to
		# ignore the class.
		num_valid_entries = tf.reduce_sum(tf.cast(
			tf.not_equal(denominator, 0), dtype=tf.float32))

		# If the value of the denominator is 0, set it to 1 to avoid
		# zero division.
		denominator = tf.where(
			tf.greater(denominator, 0),
			denominator,
			tf.ones_like(denominator))
		iou = tf.div(cm_diag, denominator)

		for i in range(params['num_classes']):
			tf.identity(iou[i], name='train_iou_class{}'.format(i))
			tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

		# If the number of valid entries is 0 (no classes) we return 0.
		result = tf.where(
			tf.greater(num_valid_entries, 0),
			tf.reduce_sum(iou, name=name) / num_valid_entries,
			0)
		return result

	sem_train_mean_iou = compute_mean_iou(sem_mean_iou[1])
	aux_sem_train_mean_iou = compute_mean_iou(aux_sem_mean_iou[1])

	tf.identity(sem_train_mean_iou, name='sem_train_mean_iou')
	tf.summary.scalar('sem_train_mean_iou', sem_train_mean_iou)

	tf.identity(aux_sem_train_mean_iou, name='aux_sem_train_mean_iou')
	tf.summary.scalar('aux_sem_train_mean_iou', aux_sem_train_mean_iou)

	return tf.estimator.EstimatorSpec(
		mode=mode,
		predictions=predictions,
		loss=loss,
		train_op=train_op,
		eval_metric_ops=metrics,
		evaluation_hooks=[]
	)
