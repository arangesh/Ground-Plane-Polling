"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import random
import threading
import warnings

import keras
from .. import backend

from ..utils.anchors import anchor_targets_bbox, bbox_transform, dim_transform
from ..utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from ..utils.transform import transform_aabb, transform_aabb_old


class Generator(object):
    """ Abstract generator class.
    """

    def __init__(
        self,
        transform_generator = None,
        batch_size=1,
        group_method='ratio',  # one of 'none', 'random', 'ratio'
        shuffle_groups=True,
        image_min_side=800,
        image_max_side=1333,
        transform_parameters=None,
        compute_anchor_targets=anchor_targets_bbox,
    ):
        """ Initialize Generator object.

        Args
            transform_generator    : A generator used to randomly transform images and annotations.
            batch_size             : The size of the batches to generate.
            group_method           : Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups         : If True, shuffles the groups each epoch.
            image_min_side         : After resizing the minimum side of an image is equal to image_min_side.
            image_max_side         : If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters   : The transform parameters used for data augmentation.
            compute_anchor_targets : Function handler for computing the targets of anchors for an image and its annotations.
        """
        self.transform_generator    = transform_generator
        self.batch_size             = int(batch_size)
        self.group_method           = group_method
        self.shuffle_groups         = shuffle_groups
        self.image_min_side         = image_min_side
        self.image_max_side         = image_max_side
        self.transform_parameters   = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets

        # create tensorflow graph and session for image transforms on the fly
        self.graph = backend.Graph()
        with self.graph.as_default():
            self.im_in = keras.backend.placeholder()
            im_in = keras.backend.cast(self.im_in, keras.backend.floatx())
            im_in = backend.random_brightness(im_in/255.0, 0.15)
            im_in = backend.random_contrast(im_in, 0.5, 1.5)
            im_in = backend.random_saturation(im_in, 0.5, 1.5)
            im_in = backend.random_hue(im_in, 0.2)
            self.im_out = 255*keras.backend.clip(im_in, 0.0, 1.0)
        self.session = backend.Session(graph = self.graph)

        self.group_index = 0
        self.lock        = threading.Lock()

        self.group_images()

    def size(self):
        """ Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """ Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """ Load annotations for all images in group.
        """
        groups = [self.load_annotations(image_index) for image_index in group]
        annotations_group, ignore_group = zip(*groups)
        return annotations_group, ignore_group

    def filter_annotations(self, image_group, annotations_group, group):
        """ Filter annotations by removing those that are outside of the image entirely or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            assert(isinstance(annotations, np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotations))

            # # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            # invalid_indices = np.where(
            #     (annotations[:, 2] <= annotations[:, 0]) |
            #     (annotations[:, 3] <= annotations[:, 1]) |
            #     (annotations[:, 0] < 0) |
            #     (annotations[:, 1] < 0) |
            #     (annotations[:, 2] > image.shape[1]) |
            #     (annotations[:, 3] > image.shape[0])
            # )[0]

            # test x2 < x1 | y2 < y1 | x2 <= 0 | y2 <= 0
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |
                (annotations[:, 3] <= annotations[:, 1]) |
                (annotations[:, 2] <= 0) |
                (annotations[:, 3] <= 0)
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    [annotations[invalid_index, :] for invalid_index in invalid_indices]
                ))
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        """ Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations, ignore_region):
        """ Randomly transforms image and annotation.
        """
        # randomly transform both image and annotations
        if self.transform_generator:
            # apply appearance transforms
            image = self.session.run(self.im_out, feed_dict={self.im_in: image})

            # apply geometric transforms
            transform = adjust_transform_for_image(next(self.transform_generator), image, self.transform_parameters.relative_translation)
            image     = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations = annotations.copy()
            # Choose correct columns to transfrom
            idx = list(range(12))
            idx.append(-1)
            for index in range(annotations.shape[0]):
                annotations[index, idx] = transform_aabb(transform, annotations[index, idx])

            # Transform the ignore regions
            ignore_region = ignore_region.copy()

            for index in range(ignore_region.shape[0]):
                ignore_region[index, :] = transform_aabb_old(transform, ignore_region[index, :])

        return image, annotations, ignore_region

    def resize_image(self, image):
        """ Resize an image using image_min_side and image_max_side.
        """
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_image(self, image):
        """ Preprocess an image (e.g. subtracts ImageNet mean).
        """
        return preprocess_image(image)

    def preprocess_group_entry(self, image, annotations, calibration, ignore_region):
        """ Preprocess image and its annotations.
        """

        # randomly transform image and annotations
        image, annotations, ignore_region = self.random_transform_group_entry(image, annotations, ignore_region)

        # preprocess the image
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :12] *= image_scale
        ignore_region *= image_scale
        calibration = np.dot(np.array([[image_scale, 0.0, 0.0], [0.0, image_scale, 0.0], [0.0, 0.0, 1.0]]), calibration)

        return image, annotations, calibration, ignore_region

    def preprocess_group(self, image_group, annotations_group, calibration_group, ignore_group):
        """ Preprocess each image and its annotations in its group.
        """
        for index, (image, annotations, calibration, ignore_region) in enumerate(zip(image_group, annotations_group, calibration_group, ignore_group)):
            # preprocess a single group entry
            image, annotations, calibration, ignore_region = self.preprocess_group_entry(image, annotations, calibration, ignore_region)

            # copy processed data back to group
            image_group[index]       = image
            annotations_group[index] = annotations
            calibration_group[index] = calibration
            ignore_group[index] = ignore_region

        return image_group, annotations_group, calibration_group, ignore_group

    def group_images(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_targets(self, image_group, annotations_group, ignore_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # compute labels and regression targets
        labels_group     = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        regression_dim_group = [None] * self.batch_size
        for index, (image, annotations, ignore_region) in enumerate(zip(image_group, annotations_group, ignore_group)):
            # compute regression targets
            labels_group[index], annotations, anchors, labels_dim, annotations_dim = self.compute_anchor_targets(
                max_shape,
                annotations,
                ignore_region,
                self.num_classes(),
                mask_shape=image.shape,
            )
            regression_group[index], regression_sign = bbox_transform(anchors, annotations, self.num_classes())
            annotations_dim = dim_transform(annotations_dim)

            # account for extra orientation classes
            anchor_states       = np.max(labels_group[index], axis=1, keepdims=True)
            labels_group[index] = np.multiply(np.concatenate([labels_group[index], labels_group[index]], axis=1), regression_sign)
            labels_group[index][anchor_states[:, 0] == -1, :] = -1

            # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
            regression_group[index]      = np.concatenate([regression_group[index], anchor_states], axis=1)
            regression_dim_group[index]  = np.append(np.reshape(annotations_dim, (-1, 3)), np.reshape(labels_dim, (-1, 1)), axis=1)

        labels_batch          = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch      = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())
        regression_dim_batch  = np.zeros((self.batch_size,) + regression_dim_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression, regression_dim) in enumerate(zip(labels_group, regression_group, regression_dim_group)):
            labels_batch[index, ...]         = labels
            regression_batch[index, ...]     = regression
            regression_dim_batch[index, ...] = regression_dim

        return [regression_batch, regression_dim_batch, labels_batch]

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        annotations_group, ignore_group = self.load_annotations_group(group)
        annotations_group = list(annotations_group)
        ignore_group = list(ignore_group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group, ignore_group = self.preprocess_group(image_group, annotations_group, ignore_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group, ignore_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)
