#!/usr/bin/env python

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

import argparse
import os
import sys
import cv2
import numpy as np

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet_3D.bin  # noqa: F401
    __package__ = "keras_retinanet_3D.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from ..preprocessing.kitti import KittiGenerator
from ..utils.transform import random_transform_generator
from ..utils.visualization import draw_annotations, draw_boxes, draw_anchors, draw_annotations_with_keypoints
from ..utils.anchors import anchors_for_shape


def create_generator(args):
    """ Create the data generators.
    Args:
        args: parseargs arguments object.
    """
    # create random transform generator for augmenting training data
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.0,
        max_shear=0.0,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.0,
    )

    if args.dataset_type == 'kitti':
        generator = KittiGenerator(
            args.kitti_path,
            subset=args.subset,
            transform_generator=transform_generator,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Debug script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')
    kitti_parser.add_argument('subset', help='Argument for loading a subset from train/val.')

    parser.add_argument('-l', '--loop', help='Loop forever, even if the dataset is exhausted.', action='store_true')
    parser.add_argument('--no-resize', help='Disable image resizing.', dest='resize', action='store_false')
    parser.add_argument('--anchors', help='Show positive anchors on the image.', action='store_true')
    parser.add_argument('--annotations', help='Show annotations on the image. Green annotations have anchors, red annotations don\'t and therefore don\'t contribute to training.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

    return parser.parse_args(args)


def run(generator, args):
    """ Main loop.
    Args
        generator: The generator to debug.
        args: parseargs args object.
    """
    # display images, one at a time
    for i in range(generator.size()):
        # load the data
        image       = generator.load_image(i)
        annotations, ignore_region = generator.load_annotations(i)

        # apply random transformations
        if args.random_transform:
            image, annotations, ignore_region = generator.random_transform_group_entry(image, annotations, ignore_region)

        # resize the image and annotations
        if args.resize:
            image, image_scale = generator.resize_image(image)
            annotations[:, :12] *= image_scale
            ignore_region *= image_scale

        # draw anchors on the image
        if args.anchors:
            labels, placeholder, anchors, labels_dim, annotations_dim = generator.compute_anchor_targets(image.shape, annotations, ignore_region, generator.num_classes())
            draw_anchors(image, anchors, labels)

        # draw annotations on the image
        if args.annotations:
            # draw annotations in red
            #draw_annotations(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)
            draw_annotations_with_keypoints(image, annotations, color=(0, 0, 255), label_to_name=generator.label_to_name)

            # draw regressed anchors in green to override most red annotations
            # result is that annotations without anchors are red, with anchors are green
            #labels, boxes, placeholder, labels_dim, annotations_dim = generator.compute_anchor_targets(image.shape, annotations, ignore_region, generator.num_classes())
            #anchor_states   = np.max(labels, axis=1) # -1 for ignore, 0 for background, 1 for object
            #draw_boxes(image, boxes[anchor_states == 1, :], (0, 255, 0))

        cv2.imshow('Image', image)
        if cv2.waitKey() == ord('q'):
            return False
    return True


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generator
    generator = create_generator(args)

    # create the display window
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    if args.loop:
        while run(generator, args):
            pass
    else:
        run(generator, args)


if __name__ == '__main__':
    main()