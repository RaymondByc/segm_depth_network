"""Utility functions for preprocessing data sets."""

from PIL import Image
import numpy as np
import tensorflow as tf

_R_MEAN = 73.158
_G_MEAN = 82.909
_B_MEAN = 72.392


# colour map
label_colours = [
	(128, 64, 128),
	(244, 35, 232),
	(70, 70, 70),
	(102, 102, 156),
	(190, 153, 153),
	(153, 153, 153),
	(250, 170, 30),
	(220, 220, 0),
	(107, 142, 35),
	(152, 251, 152),
	(70, 130, 180),
	(220, 20, 60),
	(255, 0, 0),
	(0, 0, 142),
	(0, 0, 70),
	(0, 60, 100),
	(0, 80, 100),
	(0, 0, 230),
	(119, 11, 32)
]
def disp_to_show(disp):
	return disp * 2


def decode_sem(mask, num_images=1, num_classes=21):
	"""Decode batch of segmentation masks.

	Args:
		mask: result of inference after taking argmax.
		num_images: number of images to decode from the batch.
		num_classes: number of classes to predict (including background).

	Returns:
		A batch with num_images RGB images of the same size as the input.
	"""
	n, h, w, c = mask.shape
	assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
	                          % (n, num_images)
	outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
	for i in range(num_images):
		img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
		pixels = img.load()
		for j_, j in enumerate(mask[i, :, :, 0]):
			for k_, k in enumerate(j):
				if k < num_classes:
					pixels[k_, j_] = label_colours[k]
		outputs[i] = np.array(img)
	return outputs
def decode_disp(mask, dataset='cs'):

	mask = unnormalization(mask, dataset)
	return tf.concat([mask, mask, mask], axis=3)


def normalization(image, dataset='cs'):
	image = tf.cast(image, dtype=tf.float32)
	if dataset == 'cs':
		image = tf.log(image) / tf.log(128.0 * 1.5 / 2.66)
	else:
		image = tf.log(image) / tf.log(128.0 * 1.5 / 2)

	return image

def unnormalization(image, dataset):
	image = tf.cast(image, dtype=tf.float32)
	if dataset == 'cs':
		image = image * tf.log(128.0 * 1.5 / 2.66)
		image = tf.exp(image)
	else:
		image = image * tf.log(128.0 * 1.5 / 2)
		image = tf.exp(image)
	return image


def mean_image_addition(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
	"""Adds the given means from each image channel.

	For example:
		means = [123.68, 116.779, 103.939]
		image = _mean_image_subtraction(image, means)

	Note that the rank of `image` must be known.

	Args:
		image: a tensor of size [height, width, C].
		means: a C-vector of values to subtract from each channel.

	Returns:
		the centered image.

	Raises:
		ValueError: If the rank of `image` is unknown, if `image` has a rank other
			than three or if the number of channels in `image` doesn't match the
			number of values in `means`.
	"""
	if image.get_shape().ndims != 3:
		raise ValueError('Input must be of size [height, width, C>0]')
	num_channels = image.get_shape().as_list()[-1]
	if len(means) != num_channels:
		raise ValueError('len(means) must match the number of channels')

	channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
	for i in range(num_channels):
		channels[i] += means[i]
	return tf.concat(axis=2, values=channels)


def mean_image_subtraction(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
	"""Subtracts the given means from each image channel.

	For example:
		means = [123.68, 116.779, 103.939]
		image = _mean_image_subtraction(image, means)

	Note that the rank of `image` must be known.

	Args:
		image: a tensor of size [height, width, C].
		means: a C-vector of values to subtract from each channel.

	Returns:
		the centered image.

	Raises:
		ValueError: If the rank of `image` is unknown, if `image` has a rank other
			than three or if the number of channels in `image` doesn't match the
			number of values in `means`.
	"""
	if image.get_shape().ndims != 3:
		raise ValueError('Input must be of size [height, width, C>0]')
	num_channels = image.get_shape().as_list()[-1]
	if len(means) != num_channels:
		raise ValueError('len(means) must match the number of channels')

	channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
	for i in range(num_channels):
		channels[i] -= means[i]
	return tf.concat(axis=2, values=channels)


def random_rescale_image_and_label(image, sem, disp, _HEIGHT, _WIDTH, ratio, min_scale, max_scale):
	"""Rescale an image and label with in target scale.

	Rescales an image and label within the range of target scale.

	Args:
		image: 3-D Tensor of shape `[height, width, channels]`.
		label: 3-D Tensor of shape `[height, width, 1]`.
		min_scale: Min target scale.
		max_scale: Max target scale.

	Returns:
		Cropped and/or padded image.
		If `images` was 3-D, a 3-D float Tensor of shape
		`[new_height, new_width, channels]`.
		If `labels` was 3-D, a 3-D float Tensor of shape
		`[new_height, new_width, 1]`.
	"""
	if min_scale <= 0:
		raise ValueError('\'min_scale\' must be greater than 0.')
	elif max_scale <= 0:
		raise ValueError('\'max_scale\' must be greater than 0.')
	elif min_scale >= max_scale:
		raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

	# shape = tf.shape(image)
	height = tf.to_float(_HEIGHT)
	width = tf.to_float(_WIDTH)
	scale = tf.random_uniform(
		[], minval=min_scale, maxval=max_scale, dtype=tf.float32)
	new_height = tf.to_int32(height * scale)
	new_width = tf.to_int32(width * scale)
	image = tf.image.resize_images(image, [new_height, new_width],
	                               method=tf.image.ResizeMethod.BILINEAR)
	# ?? Since label classes are integers, nearest neighbor need to be used.
	sem = tf.image.resize_images(sem, [new_height, new_width],
	                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	disp = tf.image.resize_images(disp * scale / ratio, [new_height, new_width],
	                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

	return image, sem, disp


def random_crop_or_pad_image_and_label(image, sem, disp, crop_height, crop_width, ignore_label):
	"""Crops and/or pads an image to a target width and height.

	Resizes an image to a target width and height by rondomly
	cropping the image or padding it evenly with zeros.

	Args:
		image: 3-D Tensor of shape `[height, width, channels]`.
		label: 3-D Tensor of shape `[height, width, 1]`.
		crop_height: The new height.
		crop_width: The new width.
		ignore_label: Label class to be ignored.

	Returns:
		Cropped and/or padded image.
		If `images` was 3-D, a 3-D float Tensor of shape
		`[new_height, new_width, channels]`.
	"""
	sem = sem - ignore_label  # Subtract due to 0 padding.
	sem = tf.to_float(sem)
	disp = tf.to_float(disp)
	image_height = tf.shape(image)[0]
	image_width = tf.shape(image)[1]
	image_and_label = tf.concat([image, sem, disp], axis=2)
	image_and_label_pad = tf.image.pad_to_bounding_box(
		image_and_label, 0, 0,
		tf.maximum(crop_height, image_height),
		tf.maximum(crop_width, image_width))
	image_and_label_crop = tf.random_crop(
		image_and_label_pad, [crop_height, crop_width, 5])

	image_crop = image_and_label_crop[:, :, :3]
	sem_crop = image_and_label_crop[:, :, 3:4]
	disp_crop = image_and_label_crop[:, :, 4:]
	sem_crop += ignore_label
	sem_crop = tf.to_int32(sem_crop)
	disp_crop = tf.to_int32(disp_crop)

	return image_crop, sem_crop, disp_crop


def random_flip_left_right_image_and_label(image, sem, disp):
	"""Randomly flip an image and label horizontally (left to right).

	Args:
		image: A 3-D tensor of shape `[height, width, channels].`
		label: A 3-D tensor of shape `[height, width, 1].`

	Returns:
		A 3-D tensor of the same type and shape as `image`.
		A 3-D tensor of the same type and shape as `label`.
	"""
	uniform_random = tf.random_uniform([], 0, 1.0)
	mirror_cond = tf.less(uniform_random, .5)
	image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
	sem = tf.cond(mirror_cond, lambda: tf.reverse(sem, [1]), lambda: sem)
	disp = tf.cond(mirror_cond, lambda: tf.reverse(disp, [1]), lambda: disp)
	# disp_ori = tf.cond(mirror_cond, lambda: tf.reverse(disp_ori, [1]), lambda: disp)

	return image, sem, disp


def eval_input_fn(image_filenames, _HEIGHT, _WIDTH,  sem_filenames=None, disp_filenames=None, batch_size=1):
	"""An input function for evaluation and inference.

	Args:
		image_filenames: The file names for the inferred images.
		label_filenames: The file names for the grand truth labels.
		batch_size: The number of samples per batch. Need to be 1
				for the images of different sizes.

	Returns:
		A tuple of images and labels.
	"""

	# Reads an image from a file, decodes it into a dense tensor
	def _parse_function(filename, is_label):
		if not is_label:
			image_filename, sem_filename, disp_filename = filename, None, None
		else:
			image_filename, sem_filename, disp_filename = filename

		image_string = tf.read_file(image_filename)
		image = tf.image.decode_image(image_string)
		image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
		image.set_shape([None, None, 3])
		if _HEIGHT != 1024:
			image = tf.image.resize_images(image, (_HEIGHT, _WIDTH), method=tf.image.ResizeMethod.BILINEAR)
		else:
			image.set_shape([_HEIGHT, _WIDTH, 3])

		image = mean_image_subtraction(image)

		if not is_label:
			return image
		else:
			sem_string = tf.read_file(sem_filename)
			sem = tf.image.decode_image(sem_string)
			sem = tf.to_int32(tf.image.convert_image_dtype(sem, dtype=tf.uint8))
			sem.set_shape([None, None, 1])
			if _HEIGHT != 1024:
				sem = tf.to_int32(tf.image.resize_images(sem, (_HEIGHT, _WIDTH), method=tf.image.ResizeMethod.BILINEAR))
			else:
				sem.set_shape([_HEIGHT, _WIDTH, 1])

			disp_string = tf.read_file(disp_filename)
			disp = tf.image.decode_image(disp_string)
			disp = tf.to_float(tf.image.convert_image_dtype(disp, dtype=tf.uint8))
			disp_ori = tf.to_float(tf.image.convert_image_dtype(disp, dtype=tf.uint8))
			disp.set_shape([None, None, 1])
			disp_ori.set_shape([1024, 2048, 1])

			if _HEIGHT != 1024:

				disp = tf.image.resize_images(disp/2.66, (_HEIGHT, _WIDTH), method=tf.image.ResizeMethod.BILINEAR)
				#disp_ori = tf.image.resize_images(disp/, (1024, 2048), method=tf.image.ResizeMethod.BILINEAR)

			return image, sem, disp, disp_ori



	if sem_filenames is None:
		input_filenames = image_filenames
	else:
		input_filenames = (image_filenames, sem_filenames, disp_filenames)

	dataset = tf.data.Dataset.from_tensor_slices(input_filenames)
	if sem_filenames is None:
		dataset = dataset.map(lambda x: _parse_function(x, False))
	else:
		dataset = dataset.map(lambda x, y, z: _parse_function((x, y, z), True))
	dataset = dataset.prefetch(batch_size)
	dataset = dataset.batch(batch_size)
	iterator = dataset.make_one_shot_iterator()

	if sem_filenames is None:
		images = iterator.get_next()
		labels = None
	else:
		images, sem, disp, disp_ori  = iterator.get_next()
		labels = {}
		labels['sem'] = sem
		labels['disp'] = disp
		labels['disp_ori'] = disp_ori
		print("The labels isn't NOne")

	return images, labels
