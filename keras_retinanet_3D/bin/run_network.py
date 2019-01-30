import keras
import sys
import os
import shutil

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet_3D.bin  # noqa: F401
    __package__ = "keras_retinanet_3D.bin"

from .. import models
from ..utils.image import read_image_bgr, preprocess_image, resize_image
from ..utils.visualization import draw_3d_detections_from_pose, drawdashedline, draw_detections_with_keypoints, draw_box, draw_caption

# import miscellaneous modules
import cv2
import numpy as np
import time
import scipy.io
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple script for running the network on a directory of images.')
    
    parser.add_argument('model_path',        help='Path to inference model.', type=str)
    parser.add_argument('image_dir',         help='Path to directory of input images.', type=str)
    parser.add_argument('calib_dir',         help='Path to directory of calibration files.', type=str)
    parser.add_argument('plane_params_path', help='Path to .MAT file containing road planes.', type=str)
    parser.add_argument('output_dir',        help='Path to output directory', type=str)
    parser.add_argument('--kitti',           help='Include to save results in KITTI format.', action='store_true')
    parser.add_argument('--save-images',     help='Include to save result images.', action='store_true')
    parser.add_argument('--backbone',        help='The backbone of the model to load.', default='resnet50')
    
    return parser.parse_args(args)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def load_calibration(calib_path, image_scale):
    """ Load inverse of camera projection matrix from file.
    """
    cam_id = 2
    with open(calib_path, 'r') as f:
        line = f.readlines()[cam_id]
        
    key, value = line.split(':', 1)
    P = np.array([float(x) for x in value.split()]).reshape((3, 4))
    P = np.dot(np.array([[image_scale, 0.0, 0.0], [0.0, image_scale, 0.0], [0.0, 0.0, 1.0]]), P)
    P_inv = np.linalg.pinv(P)
    return (P, P_inv)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # load retinanet model
    model = models.load_model(args.model_path, backbone_name=args.backbone)
    #print(model.summary())

    # load all road planes
    plane_params = scipy.io.loadmat(args.plane_params_path)['road_planes_database']

    # create necessary output directories
    output_dir = os.path.join(args.output_dir, os.path.basename(args.model_path)[:-3])
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'outputs'))
    os.mkdir(os.path.join(output_dir, 'outputs', 'full'))
    if args.kitti:
        os.mkdir(os.path.join(output_dir, 'outputs', 'kitti'))
    if args.save_images:
        os.mkdir(os.path.join(output_dir, 'images'))
        #os.mkdir(os.path.join(output_dir, 'images', '2D_detections'))
        #os.mkdir(os.path.join(output_dir, 'images', '3D_detections'))
        os.mkdir(os.path.join(output_dir, 'images', 'composite'))

    for j, fn in enumerate(os.listdir(args.calib_dir)):
        calib_fp = os.path.join(args.calib_dir, fn)
        image_fp = os.path.join(args.image_dir, fn.replace('.txt', '.png'))

        # load image
        raw_image = read_image_bgr(image_fp)  

        # preprocess image for network
        image = preprocess_image(raw_image)
        image, scale = resize_image(image)

        # load calibration parameters
        P, P_inv = load_calibration(calib_fp, scale)

        # construct inputs
        inputs = [np.expand_dims(image, axis=0), np.expand_dims(P_inv, axis=0), np.expand_dims(plane_params, axis=0)]

        # process image
        start = time.time()
        # run network
        boxes, dimensions, scores, labels, orientations, keypoints, keyplanes, residuals = model.predict_on_batch(inputs)[:8]
        print("Image {}: frame rate: {:.2f}".format(j, 1.0 / (time.time() - start)))

        # correct for image scale
        boxes /= scale
        P = np.dot(np.array([[1.0/scale, 0.0, 0.0], [0.0, 1.0/scale, 0.0], [0.0, 0.0, 1.0]]), P)

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > 0.05)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        max_detections = 100
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        boxes              = boxes[0, indices[scores_sort], :]
        dimensions         = dimensions[0, indices[scores_sort], :]
        scores             = scores[scores_sort]
        labels             = labels[0, indices[scores_sort]]
        orientations       = orientations[0, indices[scores_sort]]
        keypoints          = np.reshape(keypoints[0, indices[scores_sort], :, :], (-1, 12))
        keyplanes          = np.reshape(keyplanes[0, indices[scores_sort], :, :], (-1, 4))
        residuals          = residuals[0, indices[scores_sort]]

        angles = np.empty_like(dimensions)
        locations = np.empty_like(dimensions)

        # find 6dof pose
        for i in range(len(scores)):
            X_l = keypoints[i, 0:3]
            X_m = keypoints[i, 3:6]
            X_r = keypoints[i, 6:9]
            X_t = keypoints[i, 9:12]

            if orientations[i] == 0:
                x_dir = (X_m - X_l) / np.linalg.norm(X_m - X_l)
                y_dir = (X_m - X_t) / np.linalg.norm(X_m - X_t)
                z_dir = (X_r - X_m) / np.linalg.norm(X_r - X_m)

                x_dir_perp = np.cross(y_dir, z_dir)
                z_dir_perp = np.cross(x_dir, y_dir)
                diag = np.linalg.norm(X_l - X_r)
                l_sq = diag*diag - dimensions[i, 1]*dimensions[i, 1]
                if l_sq < 0:
                    l = np.Inf
                else:
                    l = np.sqrt(l_sq)

                X_l_pred = X_m - x_dir_perp * dimensions[i, 2]
                X_r_pred = X_m + z_dir_perp * dimensions[i, 1]
                X_tmp = X_r + (X_l - X_r) * dimensions[i, 1] * dimensions[i, 1] / (diag * diag)
                X_m_pred = X_tmp + np.cross((X_r - X_l) / diag, keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])) * dimensions[i, 1] * l / diag

            elif orientations[i] == 1:
                x_dir = (X_m - X_r) / np.linalg.norm(X_m - X_r)
                y_dir = (X_m - X_t) / np.linalg.norm(X_m - X_t)
                z_dir = (X_m - X_l) / np.linalg.norm(X_m - X_l)

                x_dir_perp = np.cross(y_dir, z_dir)
                z_dir_perp = np.cross(x_dir, y_dir)
                diag = np.linalg.norm(X_l - X_r)
                l_sq = diag*diag - dimensions[i, 1]*dimensions[i, 1]
                if l_sq < 0:
                    l = np.Inf
                else:
                    l = np.sqrt(l_sq)

                X_l_pred = X_m - z_dir_perp * dimensions[i, 1]
                X_r_pred = X_m - x_dir_perp * dimensions[i, 2]
                X_tmp = X_l + (X_r - X_l) * dimensions[i, 1] * dimensions[i, 1] / (diag * diag)
                X_m_pred = X_tmp + np.cross((X_r - X_l) / diag, keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])) * dimensions[i, 1] * l / diag

            elif orientations[i] == 2:
                x_dir = (X_r - X_m) / np.linalg.norm(X_r - X_m)
                y_dir = (X_m - X_t) / np.linalg.norm(X_m - X_t)
                z_dir = (X_l - X_m) / np.linalg.norm(X_l - X_m)

                x_dir_perp = np.cross(y_dir, z_dir)
                z_dir_perp = np.cross(x_dir, y_dir)
                diag = np.linalg.norm(X_l - X_r)
                l_sq = diag*diag - dimensions[i, 1]*dimensions[i, 1]
                if l_sq < 0:
                    l = np.Inf
                else:
                    l = np.sqrt(l_sq)

                X_l_pred = X_m + z_dir_perp * dimensions[i, 1]
                X_r_pred = X_m + x_dir_perp * dimensions[i, 2]
                X_tmp = X_l + (X_r - X_l) * dimensions[i, 1] * dimensions[i, 1] / (diag * diag)
                X_m_pred = X_tmp + np.cross((X_r - X_l) / diag, keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])) * dimensions[i, 1] * l / diag

            elif orientations[i] == 3:
                x_dir = (X_l - X_m) / np.linalg.norm(X_l - X_m)
                y_dir = (X_m - X_t) / np.linalg.norm(X_m - X_t)
                z_dir = (X_m - X_r) / np.linalg.norm(X_m - X_r)

                x_dir_perp = np.cross(y_dir, z_dir)
                z_dir_perp = np.cross(x_dir, y_dir)
                diag = np.linalg.norm(X_l - X_r)
                l_sq = diag*diag - dimensions[i, 1]*dimensions[i, 1]
                if l_sq < 0:
                    l = np.Inf
                else:
                    l = np.sqrt(l_sq)

                X_l_pred = X_m + x_dir_perp * dimensions[i, 2]
                X_r_pred = X_m - z_dir_perp * dimensions[i, 1]
                X_tmp = X_r + (X_l - X_r) * dimensions[i, 1] * dimensions[i, 1] / (diag * diag)
                X_m_pred = X_tmp + np.cross((X_r - X_l) / diag, keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])) * dimensions[i, 1] * l / diag

            X_all = np.stack((X_l_pred, X_m_pred, X_r_pred), axis=-1)
            X_all = np.concatenate((X_all, np.ones((1, 3))), axis= 0)
            x_all = P @ X_all
            x_all = x_all[:2, :] / x_all[2, :]

            err_x_l = np.linalg.norm(x_all[:, 0] - boxes[i, 4:6])
            err_x_m = np.linalg.norm(x_all[:, 1] - boxes[i, 6:8])
            err_x_r = np.linalg.norm(x_all[:, 2] - boxes[i, 8:10])

            if err_x_l <= err_x_r:
                if err_x_l <= err_x_m or np.isnan(err_x_m):
                    outlier = 0
                else:
                    outlier = 1
            else:
                if err_x_r <= err_x_m or np.isnan(err_x_m):
                    outlier = 2
                else:
                    outlier = 1


            if outlier == 0:
                X_m = keypoints[i, 3:6]
                X_r = keypoints[i, 6:9]
                X_t = keypoints[i, 9:12]
                if orientations[i] == 0:
                    dimensions[i, 0] = np.linalg.norm(X_t - X_m)
                    dimensions[i, 1] = np.linalg.norm(X_r - X_m)

                    y_dir = (X_m - X_t) / dimensions[i, 0]
                    z_dir = (X_r - X_m) / dimensions[i, 1]
                    x_dir = np.cross(y_dir, z_dir)

                    locations[i, :] = (X_m + X_r) / 2 - x_dir * dimensions[i, 2] / 2

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
                elif orientations[i] == 1:
                    dimensions[i, 0] = np.linalg.norm(X_t - X_m)
                    dimensions[i, 2] = np.linalg.norm(X_r - X_m)

                    x_dir = (X_m - X_r) / dimensions[i, 2]
                    y_dir = (X_m - X_t) / dimensions[i, 0]
                    z_dir = np.cross(x_dir, y_dir)

                    locations[i, :] = (X_m + X_r) / 2 - z_dir * dimensions[i, 1] / 2

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
                elif orientations[i] == 2:
                    dimensions[i, 0] = np.linalg.norm(X_t - X_m)
                    dimensions[i, 2] = np.linalg.norm(X_r - X_m)

                    x_dir = (X_r - X_m) / dimensions[i, 2]
                    y_dir = (X_m - X_t) / dimensions[i, 0]
                    z_dir = np.cross(x_dir, y_dir)

                    locations[i, :] = (X_m + X_r) / 2 + z_dir * dimensions[i, 1] / 2

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
                elif orientations[i] == 3:
                    dimensions[i, 0] = np.linalg.norm(X_t - X_m)
                    dimensions[i, 1] = np.linalg.norm(X_r - X_m)

                    y_dir = (X_m - X_t) / dimensions[i, 0]
                    z_dir = (X_m - X_r) / dimensions[i, 1]
                    x_dir = np.cross(y_dir, z_dir)

                    locations[i, :] = (X_m + X_r) / 2 + x_dir * dimensions[i, 2] / 2

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
            elif outlier == 2:
                X_l = keypoints[i, 0:3]
                X_m = keypoints[i, 3:6]
                X_t = keypoints[i, 9:12]
                if orientations[i] == 0:
                    dimensions[i, 0] = np.linalg.norm(X_t - X_m)
                    dimensions[i, 2] = np.linalg.norm(X_l - X_m)

                    x_dir = (X_m - X_l) / dimensions[i, 2]
                    y_dir = (X_m - X_t) / dimensions[i, 0]
                    z_dir = np.cross(x_dir, y_dir)

                    locations[i, :] = (X_m + X_l) / 2 + z_dir * dimensions[i, 1] / 2

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
                elif orientations[i] == 1:
                    dimensions[i, 0] = np.linalg.norm(X_t - X_m)
                    dimensions[i, 1] = np.linalg.norm(X_l - X_m)

                    y_dir = (X_m - X_t) / dimensions[i, 0]
                    z_dir = (X_m - X_l) / dimensions[i, 1]
                    x_dir = np.cross(y_dir, z_dir)

                    locations[i, :] = (X_m + X_l) / 2 - x_dir * dimensions[i, 2] / 2

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
                elif orientations[i] == 2:
                    dimensions[i, 0] = np.linalg.norm(X_t - X_m)
                    dimensions[i, 1] = np.linalg.norm(X_l - X_m)

                    y_dir = (X_m - X_t) / dimensions[i, 0]
                    z_dir = (X_l - X_m) / dimensions[i, 1]
                    x_dir = np.cross(y_dir, z_dir)

                    locations[i, :] = (X_m + X_l) / 2 + x_dir * dimensions[i, 2] / 2

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
                elif orientations[i] == 3:
                    dimensions[i, 0] = np.linalg.norm(X_t - X_m)
                    dimensions[i, 2] = np.linalg.norm(X_l - X_m)

                    x_dir = (X_l - X_m) / dimensions[i, 2]
                    y_dir = (X_m - X_t) / dimensions[i, 0]
                    z_dir = np.cross(x_dir, y_dir)

                    locations[i, :] = (X_m + X_l) / 2 - z_dir * dimensions[i, 1] / 2

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
            else:
                X_l = keypoints[i, 0:3]
                X_r = keypoints[i, 6:9]
                diag = np.linalg.norm(X_l - X_r)
                dimensions[i, 2] = np.sqrt(diag*diag - dimensions[i, 1]*dimensions[i, 1])

                locations[i, :] = (X_l + X_r) / 2

                if orientations[i] == 0:
                    X_tmp = X_r + (X_l - X_r) * dimensions[i, 1] * dimensions[i, 1] / (diag * diag)
                    X_m = X_tmp + np.cross((X_r - X_l) / diag, keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])) * dimensions[i, 1] * dimensions[i, 2] / diag
                    x_dir = (X_m - X_l) / np.linalg.norm(X_m - X_l)
                    y_dir = -keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])
                    z_dir = (X_r - X_m) / np.linalg.norm(X_r - X_m)

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
                elif orientations[i] == 1:
                    X_tmp = X_l + (X_r - X_l) * dimensions[i, 1] * dimensions[i, 1] / (diag * diag)
                    X_m = X_tmp + np.cross((X_r - X_l) / diag, keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])) * dimensions[i, 1] * dimensions[i, 2] / diag
                    x_dir = (X_m - X_r) / np.linalg.norm(X_m - X_r)
                    y_dir = -keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])
                    z_dir = (X_m - X_l) / np.linalg.norm(X_m - X_l)

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
                elif orientations[i] == 2:
                    X_tmp = X_l + (X_r - X_l) * dimensions[i, 1] * dimensions[i, 1] / (diag * diag)
                    X_m = X_tmp + np.cross((X_r - X_l) / diag, keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])) * dimensions[i, 1] * dimensions[i, 2] / diag
                    x_dir = (X_r - X_m) / np.linalg.norm(X_r - X_m)
                    y_dir = -keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])
                    z_dir = (X_l - X_m) / np.linalg.norm(X_l - X_m)

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]
                elif orientations[i] == 3:
                    X_tmp = X_r + (X_l - X_r) * dimensions[i, 1] * dimensions[i, 1] / (diag * diag)
                    X_m = X_tmp + np.cross((X_r - X_l) / diag, keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])) * dimensions[i, 1] * dimensions[i, 2] / diag
                    x_dir = (X_l - X_m) / np.linalg.norm(X_l - X_m)
                    y_dir = -keyplanes[i, :3] / np.linalg.norm(keyplanes[i, :3])
                    z_dir = (X_m - X_r) / np.linalg.norm(X_m - X_r)

                    angles[i, :] = cv2.Rodrigues(np.stack([x_dir, y_dir, z_dir], axis=-1))[0][:, 0]


        # store full results
        outputs = {'boxes': boxes[:, :4], 'keypoints': boxes[:, 4:], 'labels':labels, 'scores':scores, 'locations': locations, 'angles':angles, 'dimensions': dimensions, 'residuals': residuals}
        scipy.io.savemat(os.path.join(output_dir, 'outputs', 'full', os.path.basename(image_fp)[:-3]+'mat'), outputs)

        # store kitti results
        if args.kitti:
            with open(os.path.join(output_dir, 'outputs', 'kitti', os.path.basename(image_fp)[:-3]+'txt'), "w") as f:
                for i in range(len(scores)):
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

                    r_y = angles[i, 1] % (2*np.pi)
                    if r_y < -np.pi:
                        r_y = r_y + 2*np.pi
                    elif r_y >= np.pi:
                        r_y = r_y - 2*np.pi
                    
                    Y = np.amax(X_all[1, :])
                    
                    h = Y - np.amin(X_all[1, :])
                    
                    alpha = r_y + np.arctan2(locations[i, 2], locations[i, 0]) + 1.5*np.pi
                    alpha = alpha % (2*np.pi)
                    if alpha < -np.pi:
                        alpha = alpha + 2*np.pi
                    elif alpha >= np.pi:
                        alpha = alpha - 2*np.pi

                    f.write("Car -1 -1 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (alpha, np.maximum(boxes[i, 0], 0.0), np.maximum(boxes[i, 1], 0.0), \
                    	np.minimum(boxes[i, 2], raw_image.shape[1]), np.minimum(boxes[i, 3], raw_image.shape[0]), h, dimensions[i, 1], dimensions[i, 2], locations[i, 0], Y, locations[i, 2], r_y, scores[i]))


        # store images
        if args.save_images:
            raw_image_copy = raw_image.copy()
            draw_detections_with_keypoints(raw_image, boxes, scores, labels, orientations, score_threshold=0.4)
            draw_3d_detections_from_pose(raw_image_copy, boxes[:, :4], orientations, residuals, scores, labels, locations, angles, dimensions, P, score_threshold=0.4)
            cv2.imwrite(os.path.join(output_dir, 'images', 'composite', os.path.basename(image_fp)), np.vstack((raw_image, raw_image_copy)))

if __name__ == '__main__':
    main()
