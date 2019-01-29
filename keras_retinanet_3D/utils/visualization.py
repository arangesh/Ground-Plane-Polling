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

import cv2
import numpy as np
import math
import matplotlib
from random import shuffle

from .colors import label_color


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness=1):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)


def draw_detections(image, boxes, scores, labels, orientations, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        orientations    : 3D orientation class
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        c = colors[int(orientations[i])]
        draw_box(image, boxes[i, :], color=c)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)


def draw_detections_with_keypoints(image, boxes, scores, labels, orientations, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 12] matrix (x1, y1, x2, y2, xl, yl, xm, ym, xr, yr, xt, yt).
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        orientations    : 3D orientation class
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    up_triangle = np.array([[0, -4], [-4, 4], [4, 4]], np.int32)
    down_triangle = np.array([[0, 4], [-4, -4], [4, -4]], np.int32)
    square = np.array([[-4, -4], [4, -4], [4, 4], [-4, 4]], np.int32)

    selection = np.where(scores > score_threshold)[0]

    boxes = np.array(boxes).astype(int)
    poly_xy_all = []

    for i in selection:
        c = colors[int(orientations[i])]
        draw_box(image, boxes[i, :4], color=c)

        cv2.circle(image, (boxes[i, 4], boxes[i, 5]), 4, (0, 255, 255), thickness=1, lineType=8, shift=0)
        poly_xy_all.append(np.hstack((up_triangle[:, 0:1] + boxes[i, 6], up_triangle[:, 1:2] + boxes[i, 7])))
        poly_xy_all.append(np.hstack((square[:, 0:1] + boxes[i, 8], square[:, 1:2] + boxes[i, 9])))
        poly_xy_all.append(np.hstack((down_triangle[:, 0:1] + boxes[i, 10], down_triangle[:, 1:2] + boxes[i, 11])))

        # draw labels
        if label_to_name is None:
            caption = str(labels[i]) + ': {0:.2f}'.format(scores[i])
        else:
            caption = label_to_name(labels[i]) + ': {0:.2f}'.format(scores[i])
        draw_caption(image, boxes[i, :], caption)

    cv2.polylines(image, poly_xy_all, 1, (0, 255, 255))

def drawdashedline(img, pt1, pt2, color, thickness=2, gap=8):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if len(pts) <= 1:
        return
    s=pts[0]
    e=pts[0]
    i=0
    for p in pts:
        s=e
        e=p
        if i%2==1:
            cv2.line(img, s, e, color, thickness)
        i+=1

def draw_3d_detections(image, boxes, plane_pts, residuals, scores, labels, orientations, P, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        plane_pts       : A [N, 12] matrix (X_l, Y_l, Z_l, X_m, Y_m, Z_m, X_r, Y_r, Z_r, X_t, Y_t, Z_t).
        residuals       : A list of N residual errors.
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        orientations    : 3D orientation class
        P               : Camera calibration matrix
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]
    if len(selection) != 0:
        colors = [tuple((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / len(selection))]
        shuffle(colors)

    for i in selection:
        # draw labels
        if label_to_name is None:
            caption = str(labels[i]) + ': {0:.2f}'.format(residuals[i])
        else:
            caption = label_to_name(labels[i]) + ': {0:.2f}'.format(residuals[i])
        draw_caption(image, boxes[i, :], caption)

        c = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        thickness = 1

        # Find 3d keypoints
        if orientations[i] == 0:
            p1 = plane_pts[i, 6:9]
            p2 = plane_pts[i, 3:6]
            p3 = plane_pts[i, 0:3]
            p4 = p1 + (p3 - p2)
            p6 = plane_pts[i, 9:12]
            p5 = p6 + (p1 - p2)
            p7 = p6 + (p3 - p2)
            p8 = p7 + (p4 - p3)
        elif orientations[i] == 1:
            p1 = plane_pts[i, 3:6]
            p2 = plane_pts[i, 0:3]
            p4 = plane_pts[i, 6:9]
            p3 = p2 + (p4 - p1)
            p5 = plane_pts[i, 9:12]
            p6 = p5 + (p2 - p1)
            p8 = p5 + (p4 - p1)
            p7 = p6 + (p3 - p2)
        elif orientations[i] == 2:
            p2 = plane_pts[i, 6:9]
            p3 = plane_pts[i, 3:6]
            p4 = plane_pts[i, 0:3]
            p1 = p2 + (p4 - p3)
            p7 = plane_pts[i, 9:12]
            p6 = p7 + (p2 - p3)
            p8 = p7 + (p4 - p3)
            p5 = p6 + (p1 - p2)
        elif orientations[i] == 3:
            p1 = plane_pts[i, 0:3]
            p4 = plane_pts[i, 3:6]
            p3 = plane_pts[i, 6:9]
            p2 = p1 + (p3 - p4)
            p8 = plane_pts[i, 9:12]
            p5 = p8 + (p1 - p4)
            p7 = p8 + (p3 - p4)
            p6 = p7 + (p2 - p3)

        X_all = np.stack((p1, p2, p3, p4, p5, p6, p7, p8), axis=-1)
        X_all = np.concatenate((X_all, np.ones((1, 8))), axis= 0)
        x_all = P @ X_all
        x_all = x_all[:2, :] / x_all[2, :]
        x_all = x_all.astype(int)

        if orientations[i] == 0:
            drawdashedline(image, (x_all[0, 2], x_all[1, 2]), (x_all[0, 3], x_all[1, 3]), c, thickness)
            drawdashedline(image, (x_all[0, 3], x_all[1, 3]), (x_all[0, 7], x_all[1, 7]), c, thickness)
            cv2.line(image, (x_all[0, 7], x_all[1, 7]), (x_all[0, 6], x_all[1, 6]), c, thickness)
            cv2.line(image, (x_all[0, 6], x_all[1, 6]), (x_all[0, 2], x_all[1, 2]), c, thickness)
            drawdashedline(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 3], x_all[1, 3]), c, thickness)
            cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 2], x_all[1, 2]), c, thickness)
            cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 7], x_all[1, 7]), c, thickness)
            cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 6], x_all[1, 6]), c, thickness)
            cv2.line(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 1], x_all[1, 1]), c, thickness)
            cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 5], x_all[1, 5]), c, thickness)
            cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 4], x_all[1, 4]), c, thickness)
            cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 0], x_all[1, 0]), c, thickness)
        elif orientations[i] == 1:
            drawdashedline(image, (x_all[0, 2], x_all[1, 2]), (x_all[0, 3], x_all[1, 3]), c, thickness)
            cv2.line(image, (x_all[0, 3], x_all[1, 3]), (x_all[0, 7], x_all[1, 7]), c, thickness)
            cv2.line(image, (x_all[0, 7], x_all[1, 7]), (x_all[0, 6], x_all[1, 6]), c, thickness)
            drawdashedline(image, (x_all[0, 6], x_all[1, 6]), (x_all[0, 2], x_all[1, 2]), c, thickness)
            cv2.line(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 3], x_all[1, 3]), c, thickness)
            drawdashedline(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 2], x_all[1, 2]), c, thickness)
            cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 7], x_all[1, 7]), c, thickness)
            cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 6], x_all[1, 6]), c, thickness)
            cv2.line(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 1], x_all[1, 1]), c, thickness)
            cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 5], x_all[1, 5]), c, thickness)
            cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 4], x_all[1, 4]), c, thickness)
            cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 0], x_all[1, 0]), c, thickness)
        elif orientations[i] == 2:
            cv2.line(image, (x_all[0, 2], x_all[1, 2]), (x_all[0, 3], x_all[1, 3]), c, thickness)
            cv2.line(image, (x_all[0, 3], x_all[1, 3]), (x_all[0, 7], x_all[1, 7]), c, thickness)
            cv2.line(image, (x_all[0, 7], x_all[1, 7]), (x_all[0, 6], x_all[1, 6]), c, thickness)
            cv2.line(image, (x_all[0, 6], x_all[1, 6]), (x_all[0, 2], x_all[1, 2]), c, thickness)
            drawdashedline(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 3], x_all[1, 3]), c, thickness)
            cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 2], x_all[1, 2]), c, thickness)
            cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 7], x_all[1, 7]), c, thickness)
            cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 6], x_all[1, 6]), c, thickness)
            drawdashedline(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 1], x_all[1, 1]), c, thickness)
            cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 5], x_all[1, 5]), c, thickness)
            cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 4], x_all[1, 4]), c, thickness)
            drawdashedline(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 0], x_all[1, 0]), c, thickness)
        elif orientations[i] == 3:
            cv2.line(image, (x_all[0, 2], x_all[1, 2]), (x_all[0, 3], x_all[1, 3]), c, thickness)
            cv2.line(image, (x_all[0, 3], x_all[1, 3]), (x_all[0, 7], x_all[1, 7]), c, thickness)
            cv2.line(image, (x_all[0, 7], x_all[1, 7]), (x_all[0, 6], x_all[1, 6]), c, thickness)
            cv2.line(image, (x_all[0, 6], x_all[1, 6]), (x_all[0, 2], x_all[1, 2]), c, thickness)
            cv2.line(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 3], x_all[1, 3]), c, thickness)
            drawdashedline(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 2], x_all[1, 2]), c, thickness)
            cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 7], x_all[1, 7]), c, thickness)
            cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 6], x_all[1, 6]), c, thickness)
            drawdashedline(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 1], x_all[1, 1]), c, thickness)
            drawdashedline(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 5], x_all[1, 5]), c, thickness)
            cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 4], x_all[1, 4]), c, thickness)
            cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 0], x_all[1, 0]), c, thickness)


def draw_3d_detections_from_pose(image, boxes, orientations, residuals, scores, labels, locations, angles, dimensions, P, label_to_name=None, score_threshold=0.5):
    """ Draws detections in an image.

    # Arguments
        image           : The image to draw on.
        boxes           : A [N, 4] matrix (x1, y1, x2, y2).
        orientations    : A list of N orientation classes.
        residuals       : A list of N residual errors.
        scores          : A list of N classification scores.
        labels          : A list of N labels.
        locations       : A [N, 3] matrix with 3D positions.
        angles          : A [N, 3] matrix with 3D orientations.
        dimensions      : A [N, 3] matrix with 3D dimensions.
        P               : Camera calibration matrix.
        label_to_name   : (optional) Functor for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]
    if len(selection) != 0:
        colors = [tuple((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / len(selection))]
        shuffle(colors)

    for i in selection:
        # draw labels
        if label_to_name is None:
            caption = str(labels[i]) + ': {0:.2f}'.format(residuals[i])
        else:
            caption = label_to_name(labels[i]) + ': {0:.2f}'.format(residuals[i])
        draw_caption(image, boxes[i, :], caption)

        c = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
        thickness = 1

        # Find 3d keypoints
        h = dimensions[i, 0]
        w = dimensions[i, 1]
        l = dimensions[i, 2]
        x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
        y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
        z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])

        R = cv2.Rodrigues(angles[i, :])[0]
        X_all = np.matmul(R, np.stack([x_corners, y_corners, z_corners], axis=0))
        
        X_all[0, :] = X_all[0, :] + locations[i, 0]
        X_all[1, :] = X_all[1, :] + locations[i, 1]
        X_all[2, :] = X_all[2, :] + locations[i, 2]

        X_all = np.concatenate((X_all, np.ones((1, 8))), axis= 0)
        x_all = P @ X_all
        x_all = x_all[:2, :] / x_all[2, :]
        x_all = x_all.astype(int)

        try:
            if orientations[i] == 0:
                drawdashedline(image, (x_all[0, 2], x_all[1, 2]), (x_all[0, 3], x_all[1, 3]), c, thickness)
                drawdashedline(image, (x_all[0, 3], x_all[1, 3]), (x_all[0, 7], x_all[1, 7]), c, thickness)
                cv2.line(image, (x_all[0, 7], x_all[1, 7]), (x_all[0, 6], x_all[1, 6]), c, thickness)
                cv2.line(image, (x_all[0, 6], x_all[1, 6]), (x_all[0, 2], x_all[1, 2]), c, thickness)
                drawdashedline(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 3], x_all[1, 3]), c, thickness)
                cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 2], x_all[1, 2]), c, thickness)
                cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 7], x_all[1, 7]), c, thickness)
                cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 6], x_all[1, 6]), c, thickness)
                cv2.line(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 1], x_all[1, 1]), c, thickness)
                cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 5], x_all[1, 5]), c, thickness)
                cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 4], x_all[1, 4]), c, thickness)
                cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 0], x_all[1, 0]), c, thickness)
            elif orientations[i] == 1:
                drawdashedline(image, (x_all[0, 2], x_all[1, 2]), (x_all[0, 3], x_all[1, 3]), c, thickness)
                cv2.line(image, (x_all[0, 3], x_all[1, 3]), (x_all[0, 7], x_all[1, 7]), c, thickness)
                cv2.line(image, (x_all[0, 7], x_all[1, 7]), (x_all[0, 6], x_all[1, 6]), c, thickness)
                drawdashedline(image, (x_all[0, 6], x_all[1, 6]), (x_all[0, 2], x_all[1, 2]), c, thickness)
                cv2.line(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 3], x_all[1, 3]), c, thickness)
                drawdashedline(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 2], x_all[1, 2]), c, thickness)
                cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 7], x_all[1, 7]), c, thickness)
                cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 6], x_all[1, 6]), c, thickness)
                cv2.line(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 1], x_all[1, 1]), c, thickness)
                cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 5], x_all[1, 5]), c, thickness)
                cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 4], x_all[1, 4]), c, thickness)
                cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 0], x_all[1, 0]), c, thickness)
            elif orientations[i] == 2:
                cv2.line(image, (x_all[0, 2], x_all[1, 2]), (x_all[0, 3], x_all[1, 3]), c, thickness)
                cv2.line(image, (x_all[0, 3], x_all[1, 3]), (x_all[0, 7], x_all[1, 7]), c, thickness)
                cv2.line(image, (x_all[0, 7], x_all[1, 7]), (x_all[0, 6], x_all[1, 6]), c, thickness)
                cv2.line(image, (x_all[0, 6], x_all[1, 6]), (x_all[0, 2], x_all[1, 2]), c, thickness)
                drawdashedline(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 3], x_all[1, 3]), c, thickness)
                cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 2], x_all[1, 2]), c, thickness)
                cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 7], x_all[1, 7]), c, thickness)
                cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 6], x_all[1, 6]), c, thickness)
                drawdashedline(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 1], x_all[1, 1]), c, thickness)
                cv2.line(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 5], x_all[1, 5]), c, thickness)
                cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 4], x_all[1, 4]), c, thickness)
                drawdashedline(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 0], x_all[1, 0]), c, thickness)
            elif orientations[i] == 3:
                cv2.line(image, (x_all[0, 2], x_all[1, 2]), (x_all[0, 3], x_all[1, 3]), c, thickness)
                cv2.line(image, (x_all[0, 3], x_all[1, 3]), (x_all[0, 7], x_all[1, 7]), c, thickness)
                cv2.line(image, (x_all[0, 7], x_all[1, 7]), (x_all[0, 6], x_all[1, 6]), c, thickness)
                cv2.line(image, (x_all[0, 6], x_all[1, 6]), (x_all[0, 2], x_all[1, 2]), c, thickness)
                cv2.line(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 3], x_all[1, 3]), c, thickness)
                drawdashedline(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 2], x_all[1, 2]), c, thickness)
                cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 7], x_all[1, 7]), c, thickness)
                cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 6], x_all[1, 6]), c, thickness)
                drawdashedline(image, (x_all[0, 0], x_all[1, 0]), (x_all[0, 1], x_all[1, 1]), c, thickness)
                drawdashedline(image, (x_all[0, 1], x_all[1, 1]), (x_all[0, 5], x_all[1, 5]), c, thickness)
                cv2.line(image, (x_all[0, 5], x_all[1, 5]), (x_all[0, 4], x_all[1, 4]), c, thickness)
                cv2.line(image, (x_all[0, 4], x_all[1, 4]), (x_all[0, 0], x_all[1, 0]), c, thickness)
        except:
            pass


def draw_annotations_with_keypoints(image, annotations, color=(255, 255, 255), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 17] matrix (x1, y1, x2, y2, xl, yl, xm, ym, xr, yr, xt, yt, height, width, length, label, orientation).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    up_triangle = np.array([[0, -4], [-4, 4], [4, 4]], np.int32)
    down_triangle = np.array([[0, 4], [-4, -4], [4, -4]], np.int32)
    square = np.array([[-4, -4], [4, -4], [4, 4], [-4, 4]], np.int32)

    annotations = np.array(annotations).astype(int)
    poly_xy_all = []

    for a in annotations:
        label   = a[15]
        c       = color if color is not None else label_color(label)
        caption = '{}; {}'.format(label_to_name(label) if label_to_name else label, str(a[13]))
        draw_caption(image, a, caption)

        draw_box(image, a[:4], color=c)

        cv2.circle(image, (a[4], a[5]), 4, c, thickness=1, lineType=8, shift=0)
        poly_xy_all.append(np.hstack((up_triangle[:, 0:1] + a[6], up_triangle[:, 1:2] + a[7])))
        poly_xy_all.append(np.hstack((square[:, 0:1] + a[8], square[:, 1:2] + a[9])))
        poly_xy_all.append(np.hstack((down_triangle[:, 0:1] + a[10], down_triangle[:, 1:2] + a[11])))

    if annotations.size > 0:
        cv2.polylines(image, poly_xy_all, 1, c)

def draw_annotations(image, annotations, color=(255, 255, 255), label_to_name=None):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        annotations   : A [N, 17] matrix (x1, y1, x2, y2, xl, yl, xm, ym, xr, yr, xt, yt, height, width, length, label, orientation).
        color         : The color of the boxes. By default the color from keras_retinanet.utils.colors.label_color will be used.
        label_to_name : (optional) Functor for mapping a label to a name.
    """
    for a in annotations:
        label   = a[15]
        c       = color if color is not None else label_color(label)
        caption = '{}; {}'.format(label_to_name(label) if label_to_name else label, str(a[13]))
        draw_caption(image, a, caption)

        draw_box(image, a[:4], color=c)

def draw_anchors(image, anchors, labels):
    """ Draws annotations in an image.

    # Arguments
        image         : The image to draw on.
        anchors       : A [N, 4] matrix (x1, y1, x2, y2).
        labels        : A [N, 4*num_classes] matrix of -1s, 0s and 1s
    """
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    anchor_states    = np.max(labels, axis=1) # -1 for ignore, 0 for background, 1 for object
    anchor_state_idx = np.argmax(labels, axis=1)

    for idx, a in enumerate(anchors):
        if anchor_states[idx] == 1:
            orientation = anchor_state_idx[idx] % 4
            draw_box(image, a, color=colors[orientation])
