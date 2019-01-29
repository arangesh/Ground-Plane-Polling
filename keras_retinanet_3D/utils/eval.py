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

from __future__ import print_function

from .anchors import compute_overlap
from .visualization import draw_detections, draw_detections_with_keypoints, draw_3d_detections, draw_annotations

import numpy as np
import os
import shutil

import cv2


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=300, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 12+3(dimensions)+score+3(plane_pts)+4(planes)]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    if save_path is not None:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, '2D_detections'))
        os.mkdir(os.path.join(save_path, '3D_detections'))

    all_detections = [[None for i in range(4*generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)
        P            = np.dot(np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]]), generator.load_calibration(i))
        P_inv        = np.linalg.pinv(P)

        # run network
        inputs = [np.expand_dims(image, axis=0), np.expand_dims(P_inv, axis=0), np.expand_dims(generator.plane_params, axis=0)]
        boxes, dimensions, scores, labels, orientations, plane_pts, planes, residuals = model.predict_on_batch(inputs)[:8]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes        = boxes[0, indices[scores_sort], :]
        dimensions         = dimensions[0, indices[scores_sort], :]
        image_scores       = scores[scores_sort]
        image_labels       = labels[0, indices[scores_sort]]
        image_orientations = orientations[0, indices[scores_sort]]
        plane_pts          = plane_pts[0, indices[scores_sort], :, :]
        plane_pts          = np.reshape(plane_pts, (plane_pts.shape[0], 12))
        planes             = planes[0, indices[scores_sort], :, :]
        planes             = np.reshape(planes, (planes.shape[0], 4))
        residuals          = residuals[0, indices[scores_sort]]

        image_detections   = np.concatenate([image_boxes, dimensions, np.expand_dims(image_scores, axis=1),\
            plane_pts, planes, np.expand_dims(image_orientations, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_path is not None:
            raw_image_copy = raw_image.copy()
            #draw_annotations(raw_image, generator.load_annotations(i)[0], label_to_name=generator.label_to_name)
            #draw_detections(raw_image, image_boxes[:, :4], image_scores, image_labels, image_orientations, label_to_name=generator.label_to_name, score_threshold=0.5)
            draw_detections_with_keypoints(raw_image, image_boxes, image_scores, image_labels, image_orientations, label_to_name=generator.label_to_name, score_threshold=0.5)
            P = np.dot(np.array([[1.0/scale, 0.0, 0.0], [0.0, 1.0/scale, 0.0], [0.0, 0.0, 1.0]]), P)
            draw_3d_detections(raw_image_copy, image_boxes[:, :4], plane_pts, residuals, image_scores, image_labels, image_orientations, P, label_to_name=generator.label_to_name, score_threshold=0.5)

            cv2.imwrite(os.path.join(save_path, '2D_detections', '{}.png'.format(i)), raw_image)
            cv2.imwrite(os.path.join(save_path, '3D_detections', '{}.png'.format(i)), raw_image_copy)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            for orientation in range(4):
                all_detections[i][4*label+orientation] = image_detections[np.logical_and(image_detections[:, -1] == label, image_detections[:, -2] == orientation), :-2]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_detections, 12+3]

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(4*generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        # load the annotations
        annotations = generator.load_annotations(i)[0]

        for label in range(generator.num_classes()):
            for orientation in range(4):
                ann = annotations[np.logical_and(annotations[:, -2] == label, annotations[:, -1] == orientation), :].copy()
                all_annotations[i][4*label+orientation] = ann[:, :15]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_annotations


def evaluate(
    generator,
    model,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        model           : The model to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """
    # gather all detections and annotations
    all_detections     = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, save_path=save_path)
    all_annotations    = _get_annotations(generator)
    average_precisions = {}
    regression_errors  = []

    # process detections and annotations
    for label in range(4*generator.num_classes()):
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[15])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d[:4], axis=0), annotations[:, :4])
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                    regression_errors.append(np.absolute(d[4:15] - annotations[assigned_annotation, 4:15]))
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)

        # compute recall and precision
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision  = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    if len(regression_errors) == 0:
        keypoint_error = 0
        height_error = 0
        width_error = 0
        length_error = 0
    else:
        regression_errors = np.vstack(regression_errors)
        keypoint_error = np.average(regression_errors[:, :8], axis=None)
        height_error = np.average(regression_errors[:, 8], axis=None)
        width_error = np.average(regression_errors[:, 9], axis=None)
        length_error = np.average(regression_errors[:, 10], axis=None)

    return average_precisions, keypoint_error, height_error, width_error, length_error
