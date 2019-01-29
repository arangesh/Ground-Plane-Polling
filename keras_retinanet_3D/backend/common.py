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

import keras.backend
from .. import backend
from .dynamic import meshgrid
import numpy as np


def dim_transform_inv(dims, mean=None, std=None):
    """ Unnormalizes (usually regression results) to get predicted dims.

    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the dims.
    
    Args
        dims : np.array of shape (B, N, 3*num_classes), where B is the batch size, N the number of anchors, with num_classes (height, width, length) for each anchor.
        mean  : The mean value used when computing dims.
        std   : The standard deviation used when computing dims.

    Returns
        A np.array of the same shape as dims, but unnormalized.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """

    pred_dims = dims * std + mean

    return pred_dims


def bbox_transform_inv(boxes, deltas, sign, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.
    
    Args
        boxes : Array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for 
        (x1, y1, x2, y2). Typically anchors.
        deltas: Array of shape (B, N, 12). These deltas (d_x1, d_y1, d_x2, d_y2, d_xl, d_yl, d_xm, d_ym, d_xr, d_yr, d_xt, d_yt) 
        are a factor of the width/height.
        sign: Array of shape (B, N, 2).
        mean  : The mean value used when computing deltas.
        std   : The standard deviation used when computing deltas.

    Returns
        A np.array of shape (B, N, 12), with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height
    xl = boxes[:, :, 0] + (deltas[:, :, 4] * std[4] + mean[4]) * width
    yl = boxes[:, :, 3] + (deltas[:, :, 5] * std[5] + mean[5]) * height
    xm = (boxes[:, :, 0] + boxes[:, :, 2])/2 + backend.multiply((deltas[:, :, 6] * std[6] + mean[6]) * width, sign[:, :, 0])
    ym = boxes[:, :, 3] + (deltas[:, :, 7] * std[7] + mean[7]) * height
    xr = boxes[:, :, 2] + (deltas[:, :, 8] * std[8] + mean[8]) * width
    yr = boxes[:, :, 3] + (deltas[:, :, 9] * std[9] + mean[9]) * height
    xt = (boxes[:, :, 0] + boxes[:, :, 2])/2 + backend.multiply((deltas[:, :, 10] * std[10] + mean[10]) * width, sign[:, :, 1])
    yt = boxes[:, :, 1] + (deltas[:, :, 11] * std[11] + mean[11]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2, xl, yl, xm, ym, xr, yr, xt, yt], axis=2)

    return pred_boxes


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors
