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
#import scipy
#import sys


def anchor_targets_bbox(
    image_shape,
    annotations,
    ignore_region,
    num_classes,
    mask_shape=None,
    negative_overlap=0.4,
    positive_overlap=0.5,
    **kwargs
):
    """ Generate anchor targets for bbox detection.

    Args
        image_shape: Shape of the image.
        annotations: np.array of shape (N, 17) for (x1, y1, x2, y2, xl, yl, xm, ym, xr, yr, xt, yt, height, width, length, class, orientation).
        ignore_region: np.array of shape (M, 4) for (x1, y1, x2, y2) denoting ignore region in an image.
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    Returns
        labels: np.array of shape (A, 4*num_classes) where a cols consists of -1 for ignore, 0 for negative and 1 for positive for a certain class.
        annotations: np.array of shape (A, 12) for (x1, y1, x2, y2, xl, yl, xm, ym, xr, yr, xt, yt).
        anchors: np.array of shape (A, 4) for (x1, y1, x2, y2) containing the anchor boxes.
        labels_dim: np.array of shape (A, num_classes) where cols consists of -1 for ignore, 0 for negative and 1 for positive for a certain class.
        annotations_dim: np.array of shape (A, 3*num_classes) where cols consists of (height, width, length) for each class.
    """
    anchors = anchors_for_shape(image_shape, **kwargs)
    #scipy.io.savemat('anchors.mat', {'anchors': anchors})
    #sys.exit(0)

    if annotations.shape[0]:
        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.ones((anchors.shape[0], 4*num_classes)) * -1
        labels_dim = np.ones((anchors.shape[0], num_classes)) * -1
        # obtain indices of gt annotations with the greatest overlap
        overlaps             = compute_overlap(anchors, annotations)
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps         = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # compute box regression targets
        annotations = annotations[argmax_overlaps_inds]        

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < negative_overlap, :] = 0
        labels_dim[max_overlaps < negative_overlap, :] = 0

        # fg label: above threshold IOU
        positive_indices = max_overlaps >= positive_overlap
        labels[positive_indices, :] = 0
        labels_dim[positive_indices, :] = 0

        # retain positive indices for dimension regression
        labels_dim[positive_indices, annotations[positive_indices, -2].astype(int)] = 1
        annotations_dim = np.tile(annotations[:, 12:-2], (1, num_classes))

        # retain positive indices for only the correct 3D orientation
        class_indices = 4*annotations[positive_indices, -2] + annotations[positive_indices, -1]

        # Identify anchor associated with correct class and orientation and set as positive
        labels[positive_indices, class_indices.astype(int)] = 1

        # retain only bbox columns 
        annotations = annotations[:, :12]
    else:
        # no annotations? then everything is background
        labels_dim      = np.zeros((anchors.shape[0], num_classes))
        annotations_dim = np.zeros((anchors.shape[0], 3*num_classes))
        labels          = np.zeros((anchors.shape[0], 4*num_classes))
        annotations     = np.zeros((anchors.shape[0], 12))

    # ignore anchors inside ignore boxes
    anchors_centers    = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
    indices = np.zeros((anchors_centers.shape[0])).astype(bool)
    for region in ignore_region:
        indices = np.logical_or(indices, np.logical_and.reduce((anchors_centers[:, 0] >= region[0], anchors_centers[:, 1] >= region[1], anchors_centers[:, 0] <= region[2], anchors_centers[:, 1] <= region[3])))
    labels_dim[indices, :] = -1
    labels[indices, :] = -1

    return labels, annotations, anchors, labels_dim, annotations_dim


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)

    return shape


def make_shapes_callback(model):
    """ Make a function for getting the shape of the pyramid levels.
    """
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):
    """ Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        ratios: List of ratios with which anchors are generated (defaults to [0.5, 1, 2]).
        scales: List of scales with which anchors are generated (defaults to [2^0, 2^(1/3), 2^(2/3)]).
        strides: Stride per pyramid level, defines how the pyramids are constructed.
        sizes: Sizes of the anchors per pyramid level.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]
    if strides is None:
        strides = [2 ** x for x in pyramid_levels]
    if sizes is None:
        sizes = [2 ** (x + 2) for x in pyramid_levels]
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** (-2.0 / 3.0), 2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** (-2.0 / 3.0), 2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, num_classes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([-0.0373, -0.0165, 0.0373, 0.0171, -0.0286, -0.0478, 0.2929, 0.0114, 0.0288, -0.0589, 0.2932, -0.0007])
    if std is None:
        std = np.array([0.1957, 0.1896, 0.1957, 0.1897, 0.1967, 0.2034, 0.2046, 0.1898, 0.1964, 0.2052, 0.2048, 0.1903])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights
    targets_dxl = (gt_boxes[:, 4] - anchors[:, 0]) / anchor_widths
    targets_dyl = (gt_boxes[:, 5] - anchors[:, 3]) / anchor_heights
    targets_dxm = (gt_boxes[:, 6] - (anchors[:, 0] + anchors[:, 2])/2) / anchor_widths
    targets_dym = (gt_boxes[:, 7] - anchors[:, 3]) / anchor_heights
    targets_dxr = (gt_boxes[:, 8] - anchors[:, 2]) / anchor_widths
    targets_dyr = (gt_boxes[:, 9] - anchors[:, 3]) / anchor_heights
    targets_dxt = (gt_boxes[:, 10] - (anchors[:, 0] + anchors[:, 2])/2) / anchor_widths
    targets_dyt = (gt_boxes[:, 11] - anchors[:, 1]) / anchor_heights

    targets_sign = (np.sign(targets_dxm) + 1)/2
    targets_sign = np.concatenate((np.tile(1 - targets_sign, (4*num_classes, 1)), np.tile(targets_sign, (4*num_classes, 1))), axis = 0)
    targets_sign = targets_sign.T
    targets_dxm = np.absolute(targets_dxm)
    targets_dxt = np.absolute(targets_dxt)

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2, targets_dxl, targets_dyl, targets_dxm, targets_dym, targets_dxr, targets_dyr, targets_dxt, targets_dyt))
    targets = targets.T

    targets = (targets - mean) / std

    return targets, targets_sign

def dim_transform(gt_dims, mean=None, std=None):
    """Compute dimension regression targets for an image."""

    # mean and std should have 3*num_classes elements corresponding to (height, width, length) of each class
    if mean is None:
        mean = np.array([1.6570, 1.7999, 4.2907])
    if std is None:
        std = np.array([0.2681, 0.2243, 0.6281])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    targets = (gt_dims - mean) / std

    return targets


def compute_overlap(a, b):
    """
    Args

        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua
