"""Train a DeepLab v3 plus model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import deeplab_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug

import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default=None,
                    help='Base directory for the model.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=200,
                    help='Number of training epochs: '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                         'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                         'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                         'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                         'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=4,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=30000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--data_dir', type=str, default='/home/fanlei/aspp_branch/aspp_4_up_project_silog_loss_ffix_ratio/data/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--base_architecture', type=str, default='resnet_v2_50',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--pre_trained_model', type=str, default='./init_checkpoints/',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze batch normalization parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

parser.add_argument('--device', type=int, default=None)
parser.add_argument('--dataset', type=str, default='cs', choices=['kitti', 'cs'])
parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--aux_ratio', type=float, default=0.2)




_NUM_CLASSES = 19
_DEPTH = 3
_MIN_SCALE = 1.0
_MAX_SCALE = 1.5
_IGNORE_LABEL = 255

_POWER = 0.9
_MOMENTUM = 0.9

_BATCH_NORM_DECAY = 0.9997

_NUM_IMAGES = {
	'train': 2974,
	'validation': 500
}


def get_filenames(is_training, data_dir):
	"""Return a list of filenames.

	Args:
		is_training: A boolean denoting whether the input is for training.
		data_dir: path to the the directory containing the input data.

	Returns:
		A list of file names.
	"""
	if is_training:
		return [os.path.join(data_dir, 'cityscapes_train.record')]
	else:
		return [os.path.join(data_dir, 'cityscapes_val.record')]

# def parse_record(raw_record):
#   """Parse PASCAL image and label from a tf record."""
#   keys_to_features = {
#       'image/height':
#       tf.FixedLenFeature((), tf.int64),
#       'image/width':
#       tf.FixedLenFeature((), tf.int64),
#       'image/encoded':
#       tf.FixedLenFeature((), tf.string, default_value=''),
#       'image/format':
#       tf.FixedLenFeature((), tf.string, default_value='jpeg'),
#       'label/encoded':
#       tf.FixedLenFeature((), tf.string, default_value=''),
#       'label/format':
#       tf.FixedLenFeature((), tf.string, default_value='png'),
#   }
#
#   parsed = tf.parse_single_example(raw_record, keys_to_features)
#
#   # height = tf.cast(parsed['image/height'], tf.int32)
#   # width = tf.cast(parsed['image/width'], tf.int32)
#
#   image = tf.image.decode_image(
#       tf.reshape(parsed['image/encoded'], shape=[]), _DEPTH)
#   image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
#   image.set_shape([None, None, 3])
#
#   label = tf.image.decode_image(
#       tf.reshape(parsed['label/encoded'], shape=[]), 1)
#   label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
#   label.set_shape([None, None, 1])
#
#   return image, label

def parse_record(raw_record):
	"""Parse PASCAL image and label from a tf record."""
	keys_to_features = {
		'left/height':
			tf.FixedLenFeature((), tf.int64),
		'left/width':
			tf.FixedLenFeature((), tf.int64),
		'left/encoded':
			tf.FixedLenFeature((), tf.string, default_value=''),
		'left/format':
			tf.FixedLenFeature((), tf.string, default_value='png'),
		'semantic/encoded':
			tf.FixedLenFeature((), tf.string, default_value=''),
		'semantic/format':
			tf.FixedLenFeature((), tf.string, default_value='png'),
		'disparity/encoded':
			tf.FixedLenFeature((), tf.string, default_value=''),
		'disparity/format':
			tf.FixedLenFeature((), tf.string, default_value='png')
	}

	parsed = tf.parse_single_example(raw_record, keys_to_features)

	img = tf.image.decode_image(
		tf.reshape(parsed['left/encoded'], shape=[]), _DEPTH)
	img = tf.to_float(tf.image.convert_image_dtype(img, dtype=tf.uint8))
	img.set_shape([None, None, 3])

	semantic = tf.image.decode_image(
		tf.reshape(parsed['semantic/encoded'], shape=[]), 1)
	semantic = tf.to_int32(tf.image.convert_image_dtype(semantic, dtype=tf.uint8))
	semantic.set_shape([None, None, 1])

	disparity = tf.image.decode_image(
		tf.reshape(parsed['disparity/encoded'], shape=[]), 1)
	disparity = tf.to_float(tf.image.convert_image_dtype(disparity, dtype=tf.uint8))
	disparity.set_shape([None, None, 1])

	return img, semantic, disparity


def preprocess_image(image, sem, disp, is_training):
	"""Preprocess a single image of layout [height, width, depth]."""

	disp_ori = disp
	disp_ori.set_shape([_EVAL_HEIGHT, _EVAL_WIDTH, 1])

	if is_training:
		# Randomly scale the image and label.
		image, sem, disp = preprocessing.random_rescale_image_and_label(
			image, sem, disp, _HEIGHT, _WIDTH, _EVAL_WIDTH/_WIDTH, _MIN_SCALE, _MAX_SCALE)

		# Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
		image, sem, disp = preprocessing.random_crop_or_pad_image_and_label(
			image, sem, disp, _HEIGHT, _WIDTH, _IGNORE_LABEL)

		# Randomly flip the image and label horizontally.
		image, sem, disp = preprocessing.random_flip_left_right_image_and_label(
			image, sem, disp)

		image.set_shape([_HEIGHT, _WIDTH, 3])
		sem.set_shape([_HEIGHT, _WIDTH, 1])
		disp.set_shape([_HEIGHT, _WIDTH, 1])
	else:
		image.set_shape([None, None, 3])
		image = tf.image.resize_images(image, (_HEIGHT, _WIDTH), method=tf.image.ResizeMethod.BILINEAR)

		sem.set_shape([None, None, 1])
		sem = tf.image.resize_images(sem, (_HEIGHT, _WIDTH), method=tf.image.ResizeMethod.BILINEAR)
		sem = tf.to_int32(sem)

		disp.set_shape([None, None, 1])

		disp = tf.image.resize_images(disp, (_HEIGHT, _WIDTH), method=tf.image.ResizeMethod.BILINEAR)
		disp = tf.to_int32(disp)

	image = preprocessing.mean_image_subtraction(image)
	disp = preprocessing.normalization(disp, dataset=FLAGS.dataset)

	return image, sem, disp


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
	"""Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

	Args:
		is_training: A boolean denoting whether the input is for training.
		data_dir: The directory containing the input data.
		batch_size: The number of samples per batch.
		num_epochs: The number of epochs to repeat the dataset.

	Returns:
		A tuple of images and labels.
	"""
	dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
	dataset = dataset.flat_map(tf.data.TFRecordDataset)

	if is_training:
		# When choosing shuffle buffer sizes, larger sizes result in better
		# randomness, while smaller sizes have better performance.
		# is a relatively small dataset, we choose to shuffle the full epoch.
		dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

	dataset = dataset.map(parse_record)
	dataset = dataset.map(lambda image, sem, disp: preprocess_image(image, sem, disp, is_training))
	dataset = dataset.prefetch(batch_size)

	# We call repeat after shuffling, rather than before, to prevent separate
	# epochs from blending together.
	dataset = dataset.repeat(num_epochs)
	dataset = dataset.batch(batch_size)

	iterator = dataset.make_one_shot_iterator()
	images, sem, disp = iterator.get_next()
	labels = {}
	labels['sem'] = sem
	labels['disp'] = disp
	# labels['disp_ori'] = disp_ori

	return images, labels


def main(unused_argv):
	# Using the Winograd non-fused algorithms provides a small performance boost.
	print('The Flags:')
	print(FLAGS)
	os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
	if FLAGS.device != None:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)

	if FLAGS.clean_model_dir:
		shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

	# Set up a RunConfig to only save checkpoints once per training cycle.
	run_config = tf.estimator.RunConfig().replace(keep_checkpoint_max=1, save_checkpoints_steps=10000)
	model = tf.estimator.Estimator(
		model_fn=deeplab_model.deeplabv3_plus_model_fn,
		model_dir=FLAGS.model_dir,
		config=run_config,
		params={
			'output_stride': FLAGS.output_stride,
			'batch_size': FLAGS.batch_size,
			'base_architecture': FLAGS.base_architecture,
			'pre_trained_model': FLAGS.pre_trained_model,
			'batch_norm_decay': _BATCH_NORM_DECAY,
			'num_classes': _NUM_CLASSES,
			'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
			'weight_decay': FLAGS.weight_decay,
			'learning_rate_policy': FLAGS.learning_rate_policy,
			'num_train': _NUM_IMAGES['train'],
			'initial_learning_rate': FLAGS.initial_learning_rate,
			'max_iter': FLAGS.max_iter,
			'end_learning_rate': FLAGS.end_learning_rate,
			'power': _POWER,
			'momentum': _MOMENTUM,
			'freeze_batch_norm': FLAGS.freeze_batch_norm,
			'initial_global_step': FLAGS.initial_global_step,
			'dataset': FLAGS.dataset,
			'ratio': FLAGS.ratio,
			'aux_ratio': FLAGS.aux_ratio
		})

	for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
		tensors_to_log = {
			# 'disp_threshold_div_e3': 'disp_threshold_div_e3',
			# 'disp_threshold_div_e2': 'disp_threshold_div_e2',
			# 'disp_threshold_div_e1': 'disp_threshold_div_e1',
			# 'abs_relative_diff': 'abs_relative_diff',
			# 'squared_relative_diff': 'squared_relative_diff',
			# 'disp_threshold_1': 'disp_threshold_sub_1',
			# 'disp_threshold_3': 'disp_threshold_sub_3',
			# 'disp_threshold_5': 'disp_threshold_sub_5',
			# 'disp_mse': 'disp_mse',
			'sem_train_px_accuracy': 'sem_train_px_accuracy',
			'sem_train_mean_iou': 'sem_train_mean_iou',
			'aux_sem_train_px_accuracy': 'aux_sem_train_px_accuracy',
			'aux_sem_train_mean_iou': 'aux_sem_train_mean_iou',			
			# 'disp_berhu_loss': 'disp_berhu_loss',
			'aux_sem_loss': 'aux_sem_loss',
			'sem_loss': 'sem_loss',
			'disp_rse_linear': 'disp_rse_linear',
			'disp_rse_log': 'disp_rse_log',
			'aux_disp_rse_linear': 'aux_disp_rse_linear',
			'aux_disp_rse_log': 'aux_disp_rse_log',			
			'disp_SILog_loss': 'disp_SILog_loss',
			'aux_disp_SILog_loss': 'aux_disp_SILog_loss',
			'learning_rate': 'learning_rate'
			# 'disp_thresold_self': 'disp_thresold_self',
			# 'disp_mean_error_self': 'disp_mean_error_self'
		}

		logging_hook = tf.train.LoggingTensorHook(
			tensors=tensors_to_log, every_n_iter=10)
		train_hooks = [logging_hook]
		eval_hooks = None

		if FLAGS.debug:
			debug_hook = tf_debug.LocalCLIDebugHook()
			train_hooks.append(debug_hook)
			eval_hooks = [debug_hook]

		tf.logging.info("Start training.")
		model.train(
			input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
			hooks=train_hooks,
			# steps=1  # For debug
		)

		tf.logging.info("Start evaluation.")
		# Evaluate the model and print results
		eval_results = model.evaluate(
			# Batch size must be 1 for testing because the images' size differs
			input_fn=lambda: input_fn(False, FLAGS.data_dir, 1),
			hooks=eval_hooks,
			# steps=1  # For debug
		)
		print(eval_results)


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	FLAGS, unparsed = parser.parse_known_args()
	if FLAGS.model_dir == None:
		FLAGS.model_dir = 'model/' + FLAGS.dataset + '/' + str(FLAGS.ratio) + '/' + str(FLAGS.aux_ratio)
	# if FLAGS.seq != None:
	# 	FLAGS.model_dir += '/' + str(FLAGS.seq)
	if FLAGS.dataset == 'cs':
		_HEIGHT = 512
		_WIDTH = 1025
		_DEPTH = 3
		_EVAL_HEIGHT = 1024
		_EVAL_WIDTH = 2048
		_EVAL_DEPTH = 3
		_NUM_IMAGES = {
			'train': 2974,
			'validation': 500,
		}
	else:
		_HEIGHT = 192
		_WIDTH = 624
		_DEPTH = 3
		_EVAL_HEIGHT = 384
		_EVAL_WIDTH = 1152
		_EVAL_DEPTH = 3
		_NUM_IMAGES = {
			'train': 199,
			'validation': 1,
		}
		FLAGS.data_dir = '/home/dataset/KITTI_data_semantics/training/tf_records'
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
