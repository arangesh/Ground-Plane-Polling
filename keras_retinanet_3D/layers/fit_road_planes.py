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

import keras
from .. import backend


def poll(points, dimension, threshold):
    """
    Args
        points                : Tensor of shape (num_batch, num_dets, num_planes, 2, 3).
        dimension             : Tensor of shape (num_batch, num_dets, 1).
        threshold             : Scalar valued threshold for residual error.
    Returns
        A list of [votes, residual].
        votes is shaped (num_batch, num_dets, num_planes) where each entry contains either 0 or 1 vote.
        residual is shaped (num_batch, num_dets, num_planes) where each entry contains the corresponding residual error.
    """
    keypoint_distance = backend.norm(points[:, :, :, 0, :]-points[:, :, :, 1, :], axis=3, keep_dims=False) # (num_batch, num_dets, num_planes)
    residual = keras.backend.abs(keypoint_distance - dimension) # (num_batch, num_dets, num_planes)
    votes = backend.where(keras.backend.greater(residual, threshold), keras.backend.zeros_like(residual), keras.backend.ones_like(residual))  # (num_batch, num_dets, num_planes)
    return [votes, residual]

def calc_X_t(d_1, d_2, X_m):
    """ 
    Args
        d_1             : Tensor of shape (num_batch, num_dets, num_planes, 1, 3) storing the plane normals.
        d_2             : Tensor of shape (num_batch, num_dets, num_planes, 1, 3) storing the reprojected rays.
        X_m             : Tensor of shape (num_batch, num_dets, num_planes, 1, 3).
    Returns
        X_t is shaped (num_batch, num_dets, num_planes, 1, 3).
    """
    perp_plane = backend.cross(d_2, backend.cross(d_1, d_2)) # (num_batch, num_dets, num_planes, 1, 3)
    num = backend.matmul(perp_plane, keras.backend.permute_dimensions(X_m, (0, 1, 2, 4, 3))) # (num_batch, num_dets, num_planes, 1, 1)
    den = backend.matmul(perp_plane, keras.backend.permute_dimensions(d_1, (0, 1, 2, 4, 3))) # (num_batch, num_dets, num_planes, 1, 1)
    X_t = X_m - backend.multiply(backend.divide(num, den), d_1) # (num_batch, num_dets, num_planes, 1, 3)
    return X_t

def fit_road_planes(boxes, dimensions, orientations, P_inv, planes):
    """ Identify 3D keypoints and keyplane for each detection
    Args
        boxes                 : Tensor of shape (num_batch, num_dets, 12) containing the boxes in (x1, y1, x2, y2, xl, yl, xm, ym, xr, yr, xt, yt) format.
        dimensions            : Tensor of shape (num_batch, num_dets, 3) containing predicted (height, width, length) of object.
        orientations          : Tensor of shape (num_batch, num_dets) containing predicted orientation class.
        P_inv                 : Tensor of shape (num_batch, 4, 3) containing the pseudo-inverse of camera projection matrices.
        planes                : Tensor of shape (num_batch, num_planes, 4) consisting of different road planes. 
    Returns
        A list of [keypoints, keyplanes].
        keypoints is shaped (num_batch, num_dets, 4, 3) and consists of the 3D location of each of 4 keypoints.
        keyplanes is shaped (num_batch, num_dets, 1, 4) and contains the fitted road plane corresponding to each detection.
        residuals is shaped (num_batch, num_dets) and contains the best 'error of fit' corresponding to the keyplane.
    """
    num_batch  = keras.backend.shape(boxes)[0]
    num_dets   = keras.backend.shape(boxes)[1]
    num_planes = keras.backend.shape(planes)[1]
    heights    = backend.gather(dimensions, [0], axis=2) # (num_batch, num_dets, 1)
    widths     = backend.gather(dimensions, [1], axis=2) # (num_batch, num_dets, 1)
    lengths    = backend.gather(dimensions, [2], axis=2) # (num_batch, num_dets, 1)
    diagonals_hw = backend.norm(backend.gather(dimensions, [0, 1], axis=2), axis=2, keep_dims=True) # (num_batch, num_dets, 1)
    diagonals_wl = backend.norm(backend.gather(dimensions, [1, 2], axis=2), axis=2, keep_dims=True) # (num_batch, num_dets, 1)
    diagonals_hl = backend.norm(backend.gather(dimensions, [0, 2], axis=2), axis=2, keep_dims=True) # (num_batch, num_dets, 1)
    orientations = backend.one_hot(orientations, depth=4, dtype=keras.backend.floatx()) # (num_batch, num_dets, 4)

    # ensure all plane normals point in the correct direction and are unit norm
    direction = -keras.backend.sign(backend.gather(planes, [1], axis=2)) # (num_batch, num_planes, 1)
    planes = backend.multiply(planes, direction) # (num_batch, num_planes, 4)
    planes = backend.divide(planes, backend.norm(planes[:, :, 0:3], axis=2, keep_dims=True)) # (num_batch, num_planes, 4)

    # calculate 3D keypoints (X_l, X_m, X_r) by intersecting all rays with all planes
    x_all = keras.backend.permute_dimensions(keras.backend.reshape(boxes[:, :, 4:], (num_batch, -1, 4, 2)), (0, 1, 3, 2)) # (num_batch, num_dets, 2, 4)
    x_all = keras.backend.concatenate([x_all, keras.backend.ones((num_batch, num_dets, 1, 4))], 2) # (num_batch, num_dets, 3, 4)
    d_all = backend.matmul(keras.backend.tile(keras.backend.expand_dims(P_inv, 1), (1, num_dets, 1, 1)), x_all) # (num_batch, num_dets, 4, 4)
    d_all = backend.multiply(d_all[:, :, 0:3, :], keras.backend.sign(backend.gather(d_all[:, :, 0:3, :], [2], axis=2))) # (num_batch, num_dets, 3, 4)
    d_all = keras.backend.tile(keras.backend.expand_dims(d_all, 2), (1, 1, num_planes, 1, 1)) # (num_batch, num_dets, num_planes, 3, 4)
    planes_all = keras.backend.tile(keras.backend.expand_dims(keras.backend.expand_dims(planes, 1), 3), (1, num_dets, 1, 1, 1)) # (num_batch, num_dets, num_planes, 1, 4)
    scales_all = backend.divide(-keras.backend.expand_dims(planes_all[:, :, :, :, 3], 4), backend.matmul(planes_all[:, :, :, :, 0:3], d_all[:, :, :, :, 0:3])) # (num_batch, num_dets, num_planes, 1, 3)
    X_all = keras.backend.permute_dimensions(backend.multiply(d_all[:, :, :, :, 0:3], keras.backend.abs(scales_all)), (0, 1, 2, 4, 3)) # (num_batch, num_dets, num_planes, 3, 3)
    z_dir_check = backend.cross(backend.gather(X_all, [0], axis=3) - backend.gather(X_all, [1], axis=3), backend.gather(X_all, [2], axis=3) - backend.gather(X_all, [1], axis=3)) # (num_batch, num_dets, num_planes, 1, 3)
    z_dir_check = z_dir_check[:, :, :, 0, 1] # (num_batch, num_dets, num_planes)
    X_t = calc_X_t(planes_all[:, :, :, :, 0:3], keras.backend.permute_dimensions(backend.gather(d_all, [3], axis=4), (0, 1, 2, 4, 3)), backend.gather(X_all, [1], axis=3)) # (num_batch, num_dets, num_planes, 1, 3)
    X_all = keras.backend.concatenate([X_all, X_t], 3) # (num_batch, num_dets, num_planes, 4, 3)

    # begin polling
    threshold = 0.7 # in meters
    votes_0, residuals_0 = poll(backend.gather(X_all, [1, 3], axis=3), heights, threshold) # (num_batch, num_dets, num_planes)
    
    _dims = keras.backend.sum(backend.multiply(orientations, keras.backend.concatenate([lengths, widths, widths, lengths], 2)), axis=2, keepdims=True) # (num_batch, num_dets, 1)
    votes_1, residuals_1 = poll(backend.gather(X_all, [0, 1], axis=3), _dims, threshold) # (num_batch, num_dets, num_planes)
    
    _dims = keras.backend.sum(backend.multiply(orientations, keras.backend.concatenate([widths, lengths, lengths, widths], 2)), axis=2, keepdims=True) # (num_batch, num_dets, 1)
    votes_2, residuals_2 = poll(backend.gather(X_all, [1, 2], axis=3), _dims, threshold) # (num_batch, num_dets, num_planes)
    
    votes_3, residuals_3 = poll(backend.gather(X_all, [0, 2], axis=3), diagonals_wl, threshold) # (num_batch, num_dets, num_planes)
    
    _dims = keras.backend.sum(backend.multiply(orientations, keras.backend.concatenate([diagonals_hl, diagonals_hw, diagonals_hw, diagonals_hl], 2)), axis=2, keepdims=True) # (num_batch, num_dets, 1)
    votes_4, residuals_4 = poll(backend.gather(X_all, [0, 3], axis=3), _dims, threshold) # (num_batch, num_dets, num_planes)
    
    _dims = keras.backend.sum(backend.multiply(orientations, keras.backend.concatenate([diagonals_hw, diagonals_hl, diagonals_hl, diagonals_hw], 2)), axis=2, keepdims=True) # (num_batch, num_dets, 1)
    votes_5, residuals_5 = poll(backend.gather(X_all, [2, 3], axis=3), _dims, threshold) # (num_batch, num_dets, num_planes)

    # accumulate votes and residuals
    votes = votes_0 + votes_1 + votes_2 + votes_3 + votes_4 + votes_5 # (num_batch, num_dets, num_planes)
    residuals = residuals_0 + residuals_1 + residuals_2 + residuals_3 + residuals_4 + residuals_5 # (num_batch, num_dets, num_planes)

    # find best fit indices
    votes = votes - keras.backend.expand_dims(keras.backend.max(votes, axis=2), axis=2) # (num_batch, num_dets, num_planes)
    residuals = backend.where(keras.backend.less(votes, 0.0), 100*keras.backend.ones_like(residuals), residuals) # (num_batch, num_dets, num_planes)
    residuals = backend.where(keras.backend.less(z_dir_check, 0.0), 100*keras.backend.ones_like(residuals), residuals) # (num_batch, num_dets, num_planes)
    best_fit_indices = keras.backend.argmin(residuals, axis=2) # (num_batch, num_dets)
    
    # find best fit planes
    keyplanes = backend.map_fn(lambda x: backend.gather(x[0], x[1], axis=0), elems=[planes, best_fit_indices], dtype=keras.backend.floatx()) # (num_batch, num_dets, 4)
    keyplanes = keras.backend.expand_dims(keyplanes, 2) # (num_batch, num_dets, 1, 4)
    
    # create indices
    _range = keras.backend.tile(keras.backend.expand_dims(backend.range(keras.backend.cast(num_dets, dtype='int64')), 0), (num_batch, 1)) # (num_batch, num_dets)
    _ind = keras.backend.stack([_range, best_fit_indices], -1) # (num_batch, num_dets, 2)

    # find best fit points
    residuals = backend.map_fn(lambda x: backend.gather_nd(x[0], x[1]), elems=[residuals, _ind], dtype=keras.backend.floatx()) # (num_batch, num_dets)
    residuals = backend.divide(residuals, 6.0) # (num_batch, num_dets)    

    X_l = backend.map_fn(lambda x: backend.gather_nd(x[0], x[1]), elems=[backend.gather(X_all, [0], axis=3), _ind], dtype=keras.backend.floatx()) # (num_batch, num_dets, 1, 3)
    X_m = backend.map_fn(lambda x: backend.gather_nd(x[0], x[1]), elems=[backend.gather(X_all, [1], axis=3), _ind], dtype=keras.backend.floatx()) # (num_batch, num_dets, 1, 3)
    X_r = backend.map_fn(lambda x: backend.gather_nd(x[0], x[1]), elems=[backend.gather(X_all, [2], axis=3), _ind], dtype=keras.backend.floatx()) # (num_batch, num_dets, 1, 3)
    X_t = backend.map_fn(lambda x: backend.gather_nd(x[0], x[1]), elems=[backend.gather(X_all, [3], axis=3), _ind], dtype=keras.backend.floatx()) # (num_batch, num_dets, 1, 3)
    keypoints = keras.backend.concatenate([X_l, X_m, X_r, X_t], 2) # (num_batch, num_dets, 4, 3)

    return [keypoints, keyplanes, residuals]


class FitRoadPlanes(keras.layers.Layer):
    """ Keras layer for identifying 3D keypoints and keyplanes
    """

    def __init__(
        self,
        **kwargs
    ):
        super(FitRoadPlanes, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ 
        Args
            inputs : List of [boxes, dimensions, P_inv, planes] tensors.
        """
        boxes          = inputs[0]
        dimensions     = inputs[1]
        orientations   = inputs[2]
        P_inv          = inputs[3]
        planes         = inputs[4]

        return fit_road_planes(boxes, dimensions, orientations, P_inv, planes)

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.
        Args
            input_shape : List of input shapes [boxes (num_batch, num_dets, 12), dimensions (num_batch, num_dets, 3), orientations (num_batch, num_dets), P_inv (num_batch, 4, 3), planes (num_batch, num_planes, 4)].
        Returns
            List of tuples representing the output shapes:
            [(num_batch, num_dets, 4, 3), (num_batch, num_dets, 1, 4), (num_batch, num_dets)]
        """
        return [(input_shape[0][0], input_shape[0][1], 4, 3), (input_shape[0][0], input_shape[0][1], 1, 4), (input_shape[0][0], input_shape[0][1])]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return len(inputs) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.
        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FitRoadPlanes, self).get_config()
        return config
