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


def filter_detections(
    boxes,
    dimensions,
    classification,
    other                 = [],
    class_specific_filter = True,
    orientation_specific_filter = False,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 100,
    nms_threshold         = 0.5
):
    """ Filter detections using the boxes and classification values.
    Args
        boxes                 : Tensor of shape (num_boxes, 12) containing the boxes in (x1, y1, x2, y2, xl, yl, xm, ym, xr, yr, xt, yt) format.
        dimensions            : Tensor of shape (num_boxes, 3*num_classes) containing the (height, width, length) of the 3D box for each class.
        classification        : Tensor of shape (num_boxes, 8*num_classes) containing the classification scores.
        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        orientation_specific_filter : Whether to perform filtering per orientation, or take the best scoring one and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
    Returns
        A list of [boxes, scores, labels, orientations, other[0], other[1], ...].
        boxes is shaped (max_detections, 12) and contains the (x1, y1, x2, y2, xl, yl, xm, ym, xr, yr, xt, yt) of the non-suppressed boxes.
        dimensions is shaped (max_detections, 3) and contains the (height, width, length) of each object.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        orientations is shaped (max_detections,) and contains the predicted orientation.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels, orientations, dimensions):
        # threshold based on score
        indices = backend.where(keras.backend.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = backend.gather_nd(boxes, indices)
            filtered_scores = backend.gather(scores, indices)[:, 0]          

            # perform NMS
            nms_indices = backend.non_max_suppression(filtered_boxes[:, :4], filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = backend.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = backend.gather_nd(labels, indices)
        orientations = backend.gather_nd(orientations, indices)
        dimensions = backend.gather_nd(dimensions, indices)
        indices = keras.backend.stack([indices[:, 0], labels, orientations], axis = 1)

        return indices, dimensions

    num_boxes = keras.backend.shape(boxes)[0]
    num_classes = 1

    # Create deep copy of classification
    half_size = keras.backend.cast(keras.backend.shape(classification)[1] / 2, dtype='int64')

    classification = keras.backend.stack([classification[:, :half_size], classification[:, half_size:]], 0) # (2, N, 4*C)
    classification = keras.backend.max(classification, axis = 0) # (N, 4*C)
    classification = keras.backend.stack([classification[:, 0::4], classification[:, 1::4], classification[:, 2::4], classification[:, 3::4]], 1) # (N, 4, C)
    classification = keras.backend.reshape(classification, (-1, num_classes))
    classification_all = classification

    if orientation_specific_filter:
        if class_specific_filter:
            all_indices = []
            all_dims    = []
            for o in range(4):
                # perform per class filtering
                for c in range(int(classification.shape[1])):
                    scores = classification[o::4, c]
                    labels = c * keras.backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
                    dims = dimensions[:, 3*c:3*(c+1)]
                    orientations = o * keras.backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
                    indices, dims = _filter_detections(scores, labels, orientations, dims)
                    all_indices.append(indices)
                    all_dims.append(dims)

            # concatenate indices to single tensor
            indices = keras.backend.concatenate(all_indices, axis = 0)
            dims    = keras.backend.concatenate(all_dims, axis = 0)
        else:
            all_indices = []
            all_dims    = []
            for o in range(4):
                scores  = keras.backend.max(classification[o::4, :], axis = 1)
                labels  = keras.backend.argmax(classification[o::4, :], axis = 1)
                dims1 = backend.gather_nd(dimensions, keras.backend.stack([backend.range(keras.backend.cast(num_boxes, dtype='int64')), 3*labels], 1))
                dims2 = backend.gather_nd(dimensions, keras.backend.stack([backend.range(keras.backend.cast(num_boxes, dtype='int64')), 3*labels+1], 1))
                dims3 = backend.gather_nd(dimensions, keras.backend.stack([backend.range(keras.backend.cast(num_boxes, dtype='int64')), 3*labels+2], 1))
                dims  = keras.backend.stack([dims1, dims2, dims3], 1)
                orientations = o * keras.backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
                indices, dims = _filter_detections(scores, labels, orientations, dims)
                all_indices.append(indices)
                all_dims.append(dims)

            # concatenate indices to single tensor
            indices = keras.backend.concatenate(all_indices, axis = 0)
            dims    = keras.backend.concatenate(all_dims, axis = 0)
    else:
        classification  = keras.backend.stack([classification[0::4, :], classification[1::4, :], classification[2::4, :], classification[3::4, :]], 0)
        orientations_all = keras.backend.argmax(classification, axis = 0)
        classification = keras.backend.max(classification, axis = 0)

        if class_specific_filter:
            all_indices = []
            all_dims    = []
            # perform per class filtering
            for c in range(int(classification.shape[1])):
                scores = classification[:, c]
                labels = c * keras.backend.ones((keras.backend.shape(scores)[0],), dtype='int64')
                dims = dimensions[:, 3*c:3*(c+1)]
                orientations = orientations_all[:, c]
                indices, dims = _filter_detections(scores, labels, orientations, dims)
                all_indices.append(indices)
                all_dims.append(dims)

            # concatenate indices to single tensor
            indices = keras.backend.concatenate(all_indices, axis = 0)
            dims    = keras.backend.concatenate(all_dims, axis = 0)
        else:
            scores  = keras.backend.max(classification, axis = 1)
            labels  = keras.backend.argmax(classification, axis = 1)
            dims1 = backend.gather_nd(dimensions, keras.backend.stack([backend.range(keras.backend.cast(num_boxes, dtype='int64')), 3*labels], 1))
            dims2 = backend.gather_nd(dimensions, keras.backend.stack([backend.range(keras.backend.cast(num_boxes, dtype='int64')), 3*labels+1], 1))
            dims3 = backend.gather_nd(dimensions, keras.backend.stack([backend.range(keras.backend.cast(num_boxes, dtype='int64')), 3*labels+2], 1))
            dims  = keras.backend.stack([dims1, dims2, dims3], 1)
            orientations = backend.gather_nd(orientations_all, keras.backend.stack([backend.range(keras.backend.cast(num_boxes, dtype='int64')), labels], 1))
            indices, dims = _filter_detections(scores, labels, orientations, dims)


    # select top k
    scores              = backend.gather_nd(classification_all, keras.backend.stack([4*indices[:, 0] + indices[:, 2], indices[:, 1]], 1))
    labels              = indices[:, 1]
    orientations        = indices[:, 2]
    dimensions          = dims
    scores, top_indices = backend.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = backend.gather(indices[:, 0], top_indices)
    boxes               = backend.gather(boxes, indices)
    labels              = backend.gather(labels, top_indices)
    dimensions          = backend.gather(dimensions, top_indices)
    orientations        = backend.gather(orientations, top_indices)
    other_              = [backend.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size       = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes          = backend.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores         = backend.pad(scores, [[0, pad_size]], constant_values=-1)
    labels         = backend.pad(labels, [[0, pad_size]], constant_values=-1)
    labels         = keras.backend.cast(labels, 'int32')
    dimensions     = backend.pad(dimensions, [[0, pad_size], [0, 0]], constant_values=-1)
    orientations   = backend.pad(orientations, [[0, pad_size]], constant_values=-1)
    orientations   = keras.backend.cast(orientations, 'int32')
    other_         = [backend.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 12])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    dimensions.set_shape([max_detections, 3])
    orientations.set_shape([max_detections])
    for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, dimensions, scores, labels, orientations] + other_


class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        nms                   = True,
        class_specific_filter = True,
        orientation_specific_filter = False,
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 100,
        parallel_iterations   = 32,
        **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.
        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            orientation_specific_filter : Whether to perform filtering per orientation, or take the best scoring one and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.orientation_specific_filter = orientation_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.
        Args
            inputs : List of [boxes, dimensions, classification, other[0], other[1], ...] tensors.
        """
        boxes          = inputs[0]
        dimensions     = inputs[1]
        classification = inputs[2]
        other          = inputs[3:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes          = args[0]
            dimensions     = args[1]
            classification = args[2]
            other          = args[3]

            return filter_detections(
                boxes,
                dimensions,
                classification,
                other,
                nms                   = self.nms,
                class_specific_filter = self.class_specific_filter,
                orientation_specific_filter = self.orientation_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = backend.map_fn(
            _filter_detections,
            elems=[boxes, dimensions, classification, other],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), keras.backend.floatx(), 'int32', 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.
        Args
            input_shape : List of input shapes [boxes, dimensions, classification, other[0], other[1], ...].
        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_dimensions.shape, filtered_scores.shape, filtered_labels.shape, filtered_orientations.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 12),
            (input_shape[1][0], self.max_detections, 3),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(3, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 2) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.
        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                   : self.nms,
            'class_specific_filter' : self.class_specific_filter,
            'orientation_specific_filter' : self.orientation_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations,
        })

        return config