"""Run inference a DeepLab v3 model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import deeplab_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc  
from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default=None,
                    help='The directory containing the image data.')

parser.add_argument('--output_dir', type=str, default=None,
                    help='Path to the directory to generate the inference results')

parser.add_argument('--infer_data_list', type=str, default=None,
                    help='Path to the file listing the inferring images.')

parser.add_argument('--model_dir', type=str, default=None,
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default='resnet_v2_50',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')
parser.add_argument('--device', type=int, default=None)
parser.add_argument('--dataset', type=str, default='cs', choices=['kitti', 'cs'])
parser.add_argument('--ratio', type=float)
parser.add_argument('--type', type=str, choices=['sem', 'disp'])
parser.add_argument('--aux_ratio', type=float)

_NUM_CLASSES = 19


def main(unused_argv):
	# Using the Winograd non-fused algorithms provides a small performance boost.
	os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
	if FLAGS.device != None:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)

	pred_hooks = None
	if FLAGS.debug:
		debug_hook = tf_debug.LocalCLIDebugHook()
		pred_hooks = [debug_hook]

	model = tf.estimator.Estimator(
		model_fn=deeplab_model.deeplabv3_plus_model_fn,
		model_dir=FLAGS.model_dir,
		params={
			'output_stride': FLAGS.output_stride,
			'batch_size': 1,  # Batch size must be 1 because the images' size may differ
			'base_architecture': FLAGS.base_architecture,
			'pre_trained_model': None,
			'batch_norm_decay': None,
			'num_classes': _NUM_CLASSES,
			'dataset': FLAGS.dataset,
			'ratio': FLAGS.ratio,
			'aux_ratio': FLAGS.aux_ratio,
		})

	examples = dataset_util.read_examples_list(FLAGS.infer_data_list)
	image_files = [os.path.join(FLAGS.data_dir, filename) for filename, sem, disp in examples]

	predictions = model.predict(
		input_fn=lambda: preprocessing.eval_input_fn(image_files, _HEIGHT, _WIDTH),
		hooks=pred_hooks)

	output_dir = FLAGS.output_dir + str(FLAGS.ratio)
	#

	if not os.path.exists(output_dir + '/sem_' + str(FLAGS.aux_ratio)):
		os.makedirs(output_dir + '/sem_' + str(FLAGS.aux_ratio))
		os.makedirs(output_dir + '/disp_' + str(FLAGS.aux_ratio))
		os.makedirs(output_dir + '/all_' + str(FLAGS.aux_ratio))

	for pred_dict, image_path in zip(predictions, image_files):
		image_basename = os.path.splitext(os.path.basename(image_path))[0]

		print("generating:", image_basename)
		img_out = pred_dict['image']
		sem_out = pred_dict['sem_decoded']
		classes_out = pred_dict['classes']
		aux_classes_out = pred_dict['aux_classes']
		disp_out = pred_dict['disp_decoded']
		aux_sem_out = pred_dict['aux_sem_decoded']
		aux_disp_out = pred_dict['aux_disp_decoded']

		# Image.fromarray(img_out).save(os.path.join(output_dir + '/img', image_basename) + '.png')
		if FLAGS.aux_ratio == 1:
			disp_img = Image.fromarray(aux_disp_out)
		else:
			disp_img = Image.fromarray(disp_out)
		disp_img.save(os.path.join(output_dir + '/disp'+ '_' + str(FLAGS.aux_ratio) + '/' + image_basename  + '.png'))
		# Image.fromarray(disp_out).resize((1226, 370)).save(
		# 	os.path.join(output_dir + '/disp_full_size', image_basename) + '.png')
		class_name = os.path.join(output_dir + '/sem' + '_' + str(FLAGS.aux_ratio) + '/' + image_basename.replace('_leftImg8bit', '_trainId') + '.png')
		if FLAGS.aux_ratio == 1:
			# sem_img = Image.fromarray(aux_sem_out)
			# sem_classes = Image.fromarray(aux_classes_out, "P")
			aux_classes_out = np.squeeze(aux_classes_out)
			misc.imsave(class_name, aux_classes_out)
		else:
			# sem_classes = Image.fromarray(classes_out)
			classes_out = np.squeeze(classes_out)

			misc.imsave(class_name, classes_out)
		# sem_img.save(os.path.join(output_dir + '/sem' + '_' + str(FLAGS.aux_ratio) + '/' + image_basename + '.png'))
		#sem_classes.save()
		# Image.fromarray(sem_out).resize((1226, 370)).save(
		# 	os.path.join(output_dir + '/sem_full_size', image_basename) + '.png')
		# disp_img = Image.open('/home/fanlei/final_version/aspp_4_up_project_silog_loss_ffix_ratio/result/kitti/05/0.0/disp_train_size/' + image_basename + '.png')
		# disp_img = np.array(disp_img, dtype=np.uint8)
		# Image.fromarray(np.concatenate((img_out, sem_img, disp_img))).save(os.path.join(output_dir + '/all_' + str(FLAGS.aux_ratio) + '/' + image_basename) + '.png')



if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)


	FLAGS, unparsed = parser.parse_known_args()

	if FLAGS.dataset == 'kitti':
		FLAGS.data_dir = '/home/dataset/KITTI_odometry/data_odometry_color/dataset/sequences'
		FLAGS.infer_data_list = '/home/dataset/KITTI_odometry/data_odometry_color/dataset/sequences/05_list.txt'
		_EVAL_HEIGHT = 384
		_EVAL_WIDTH = 1152
		_HEIGHT = 192
		_WIDTH = 624
	elif FLAGS.dataset == 'cs':
		_HEIGHT = 512
		_WIDTH = 1024
		FLAGS.data_dir = '/home/data/cityscapes/'
		FLAGS.infer_data_list = '/home/fanlei/aspp_branch/aspp_4_up_project_silog_loss_ffix_ratio/data/val.txt'
		_EVAL_HEIGHT = 1024
		_EVAL_WIDTH = 2048


	FLAGS.model_dir = '/home/fanlei/aspp_branch/aspp_4_up_project_silog_loss_ffix_ratio/model/cs/0.85/' + str(FLAGS.aux_ratio)

	if FLAGS.output_dir == None:
		FLAGS.output_dir = 'result/' + FLAGS.dataset +'/'

	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
