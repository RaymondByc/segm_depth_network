"""Converts PASCAL dataset to TFRecords file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import sys

import PIL.Image
import numpy as np
import tensorflow as tf

from utils import dataset_util

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/home/data/cityscapes',
                    help='Path to the directory containing the cityscapes data.')

parser.add_argument('--output_path', type=str, default='/home/fanlei/aspp_branch/aspp_4_up_project_silog_loss_ffix_ratio/data/',
                    help='Path to the directory to create TFRecords outputs.')

parser.add_argument('--train_data_list', type=str, default='/home/fanlei/aspp_branch/aspp_4_up_project_silog_loss_ffix_ratio/data/train.txt',
                    help='Path to the file listing the training data.')

parser.add_argument('--valid_data_list', type=str, default='/home/fanlei/aspp_branch/aspp_4_up_project_silog_loss_ffix_ratio/data/val.txt',
                    help='Path to the file listing the validation data.')

parser.add_argument('--left_data_dir', type=str, default='leftImg8bit')

parser.add_argument('--right_data_dir', type=str, default='rightImg8bit')

parser.add_argument('--semantic_data_dir', type=str, default='gtFine')

parser.add_argument('--disparity_data_dir', type=str, default='disparity')


def dict_to_tf_example(left_path, semantic_path, disparity_path):
  """Convert image and label to tf.Example proto.

  Args:
    image_path: Path to a single PASCAL image.
    label_path: Path to its corresponding label.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by image_path is not a valid JPEG or
                if the label pointed to by label_path is not a valid PNG or
                if the size of image does not match with that of label.
  """
  with tf.gfile.GFile(left_path, 'rb') as fid:
    left_png = fid.read()
  left_png_io = io.BytesIO(left_png)
  left = PIL.Image.open(left_png_io)


  with tf.gfile.GFile(semantic_path, 'rb') as fid:
    semantic_png = fid.read()
  semantic_png_io = io.BytesIO(semantic_png)
  semantic = PIL.Image.open(semantic_png_io)
  
  with tf.gfile.GFile(disparity_path, 'rb') as fid:
    disparity_png = fid.read()
  disparity_png_io = io.BytesIO(disparity_png)
  disparity = PIL.Image.open(disparity_png_io)

  if left.format != 'PNG' or  semantic.format != 'PNG' or disparity.format != 'PNG':
    raise ValueError('The format not PNG')

  if left.size != disparity.size or left.size != semantic.size:
    raise ValueError('The size of image does not match with that of label.')

  width, height = left.size

  example = tf.train.Example(features=tf.train.Features(feature={
    'left/height': dataset_util.int64_feature(height),
    'left/width': dataset_util.int64_feature(width),
    'left/encoded': dataset_util.bytes_feature(left_png),
    'left/format': dataset_util.bytes_feature('png'.encode('utf8')),
    'semantic/encoded': dataset_util.bytes_feature(semantic_png),
    'semantic/format': dataset_util.bytes_feature('png'.encode('utf8')),
    'disparity/encoded': dataset_util.bytes_feature(disparity_png),
    'disparity/format': dataset_util.bytes_feature('png'.encode('utf8'))
  }))
  return example


def create_tf_record(output_filename,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    image_dir: Directory where image files are stored.
    label_dir: Directory where label files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 500 == 0:
      tf.logging.info('On image %d of %d', idx, len(examples))
    left_path = os.path.join(FLAGS.data_dir, example[0])
    semantic_path = os.path.join(FLAGS.data_dir, example[1])
    disparity_path = os.path.join(FLAGS.data_dir, example[2])

    # zoom disparity
    # disparity_img = np.array(PIL.Image.open(disparity_path), dtype=np.uint8)
    # PIL.Image.fromarray(disparity_img).save()
    if not os.path.exists(left_path):
      tf.logging.warning('Could not find %s, ignoring example.', left_path)
      continue
    elif not os.path.exists(semantic_path):
      tf.logging.warning('Could not find %s, ignoring example.', semantic_path)
      continue
    elif not os.path.exists(disparity_path):
      tf.logging.warning('Could not find %s, ignoring example.', disparity_path)
      continue

    try:
      tf_example = dict_to_tf_example(left_path, semantic_path, disparity_path)
      writer.write(tf_example.SerializeToString())
    except ValueError:
      tf.logging.warning('Invalid example: %s, ignoring.', example)

  writer.close()


def main(unused_argv):
  if not os.path.exists(FLAGS.output_path):
    os.makedirs(FLAGS.output_path)

  tf.logging.info("Reading from cityscapes dataset")
  # left_dir = os.path.join(FLAGS.data_dir, FLAGS.left_data_dir)
  # right_dir = os.path.join(FLAGS.data_dir, FLAGS.right_data_dir)
  # semantic_dir = os.path.join(FLAGS.data_dir, FLAGS.semantic_data_dir)
  # disparity_dir = os.path.join(FLAGS.data_dir, FLAGS.disparity_data_dir)

  train_examples = dataset_util.read_examples_list(FLAGS.train_data_list)
  val_examples = dataset_util.read_examples_list(FLAGS.valid_data_list)

  train_output_path = os.path.join(FLAGS.output_path, 'cityscapes_train.record')
  val_output_path = os.path.join(FLAGS.output_path, 'cityscapes_val.record')

  create_tf_record(train_output_path, train_examples)
  create_tf_record(val_output_path, val_examples)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
