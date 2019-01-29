"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

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

import csv
import os.path

import keras
import numpy as np
import scipy.io
from PIL import Image

from .generator import Generator
from ..utils.image import read_image_bgr

kitti_classes = {
    'Car': 0,
    'Van': 0,
    #'Truck': 1,
    #'Pedestrian': 2,
    #'Person_sitting': 3,
    #'Cyclist': 4,
}


class KittiGenerator(Generator):
    """ Generate data for a KITTI dataset.

    See http://www.cvlibs.net/datasets/kitti/ for more information.
    """

    def __init__(
        self,
        base_dir,
        subset='train',
        **kwargs
    ):
        """ Initialize a KITTI data generator.

        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            subset: The subset to generate data for (defaults to 'train').
        """
        self.base_dir = base_dir
        self.subset = subset

        label_dir         = os.path.join(self.base_dir, subset, 'labels')
        image_dir         = os.path.join(self.base_dir, subset, 'images')
        calib_dir         = os.path.join(self.base_dir, subset, 'calibs')
        plane_params_path = os.path.join(self.base_dir, 'road_planes_dataset.mat')

        """
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                             'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                             'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    2D bbox      2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
        6    3D bbox      3D bounding box points of interest i.e xl, yl, xm, ym, xr, yr
        1    orientation   3D orientation class
        """

        self.id_to_labels = {}
        for label, id in kitti_classes.items():
            self.id_to_labels[id] = label

        self.image_data = dict()
        self.ignore_regions = dict()
        self.images = []
        self.calibs = []
        self.plane_params = scipy.io.loadmat(plane_params_path)['road_planes_dataset']
        for i, fn in enumerate(os.listdir(image_dir)):
            image_fp = os.path.join(image_dir, fn)
            label_fp = os.path.join(label_dir, fn.replace('.png', '.txt').replace('.jpg', '.txt'))
            calib_fp = os.path.join(calib_dir, fn.replace('.png', '.txt').replace('.jpg', '.txt'))

            self.images.append(image_fp)
            self.calibs.append(calib_fp)

            fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'left', 'top', 'right', 'bottom', 'xl', 'yl', 'xm',
                          'ym', 'xr', 'yr', 'xt', 'yt', 'height', 'width', 'length', 'orientation']
            with open(label_fp, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
                boxes = []
                ignore_region = []
                for line, row in enumerate(reader):
                    label = row['type']
                    if label == 'DontCare' or label == 'Misc':
                    	ignore_region.append({'x1': row['left'], 'x2': row['right'], 'y2': row['bottom'], 'y1': row['top']})
                    	continue
                    elif label not in kitti_classes.keys():
                    	continue
                    cls_id = kitti_classes[label]

                    annotation = {'cls_id': cls_id, 'x1': row['left'], 'x2': row['right'], 'y2': row['bottom'], 'y1': row['top'],
                                  'xl': row['xl'], 'yl': row['yl'], 'xm': row['xm'], 'ym': row['ym'], 'xr': row['xr'], 'yr': row['yr'],
                                  'xt': row['xt'], 'yt': row['yt'], 'height' : row['height'], 'width' : row['width'], 'length' : row['length'], 'orientation': row['orientation']}
                    boxes.append(annotation)

                self.image_data[i] = boxes
                self.ignore_regions[i] = ignore_region

        super(KittiGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.images)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(kitti_classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        raise NotImplementedError()

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.id_to_labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.images[image_index])
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        return read_image_bgr(self.images[image_index])

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        annotations = self.image_data[image_index]
        boxes = np.zeros((len(annotations), 17))
        for idx, ann in enumerate(annotations):
            boxes[idx, 0]  = float(ann['x1'])
            boxes[idx, 1]  = float(ann['y1'])
            boxes[idx, 2]  = float(ann['x2'])
            boxes[idx, 3]  = float(ann['y2'])
            boxes[idx, 4]  = float(ann['xl'])
            boxes[idx, 5]  = float(ann['yl'])
            boxes[idx, 6]  = float(ann['xm'])
            boxes[idx, 7]  = float(ann['ym'])
            boxes[idx, 8]  = float(ann['xr'])
            boxes[idx, 9]  = float(ann['yr'])
            boxes[idx, 10] = float(ann['xt'])
            boxes[idx, 11] = float(ann['yt'])
            boxes[idx, 12] = float(ann['height'])
            boxes[idx, 13] = float(ann['width'])
            boxes[idx, 14] = float(ann['length'])
            boxes[idx, 15] = int(ann['cls_id'])
            boxes[idx, 16] = int(ann['orientation'])

        ignore_regions = self.ignore_regions[image_index]
        ignore_boxes = np.zeros((len(ignore_regions), 4))
        for idx, region in enumerate(ignore_regions):
            ignore_boxes[idx, 0] = float(region['x1'])
            ignore_boxes[idx, 1] = float(region['y1'])
            ignore_boxes[idx, 2] = float(region['x2'])
            ignore_boxes[idx, 3] = float(region['y2'])
        return boxes, ignore_boxes

    def load_calibration(self, image_index):
        """ Load calibration at the image_index.
        """
        cam_id = 2
        with open(self.calibs[image_index], 'r') as f:
            line = f.readlines()[cam_id]
            
        key, value = line.split(':', 1)
        P = np.array([float(x) for x in value.split()]).reshape((3, 4))

        return P

    def load_calibration_group(self, group):
        """ Load calibrations for all images in group.
        """
        return [self.load_calibration(image_index) for image_index in group]

    def compute_inputs(self, image_group, calib_group):
        """ Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        P     = np.array(calib_group)
        P_inv = np.linalg.pinv(P)
        
        inputs = [image_batch, P_inv, np.tile(self.plane_params, (self.batch_size, 1, 1))]
        if self.subset == 'train':
            inputs = inputs[0]
        return inputs

    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        calib_group       = self.load_calibration_group(group)
        annotations_group, ignore_group = self.load_annotations_group(group)
        annotations_group = list(annotations_group)
        ignore_group = list(ignore_group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group, calib_group, ignore_group = self.preprocess_group(image_group, annotations_group, calib_group, ignore_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group, calib_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group, ignore_group)

        return inputs, targets
