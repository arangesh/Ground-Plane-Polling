import keras
import sys
import os
import shutil

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet_3D.bin  # noqa: F401
    __package__ = "keras_retinanet_3D.bin"

# import miscellaneous modules
import cv2
import numpy as np
import time
import scipy.io
import argparse
import h5py

from .. import models
from ..utils.image import read_image_bgr, preprocess_image

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


# camera setup
camera_ids = ['front', 'front-right', 'rear-right', 'rear', 'rear-left', 'front-left']


def resize_image(img, min_side=1000, max_side=1333):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple script for running the network on a directory of images.')
    
    parser.add_argument('model_path',        help='Path to inference model.', type=str)
    parser.add_argument('drive_path',        help='Path to drive directory', type=str)
    parser.add_argument('calib_path',        help='Path to calibration file', type=str)
    parser.add_argument('plane_params_path', help='Path to road planes dataset file', type=str)
    parser.add_argument('--backbone',        help='The backbone of the model to load.', default='vgg19')
    
    return parser.parse_args(args)

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # load retinanet model
    model = models.load_model(args.model_path, backbone_name=args.backbone)
    #print(model.summary())

    # load ground plane database, calibration parameters and videos
    plane_params = []
    P       = []
    P_inv   = []
    vids = []
    scales = [1333/1920, 1333/1920, 1333/1920, 1333/1920, 1333/1920, 1333/1920]
    for camno, camid in enumerate(camera_ids):
        plane_params.append(scipy.io.loadmat(args.plane_params_path)['road_planes_dataset'+str(camno)])
        P_tmp = scipy.io.loadmat(args.calib_path)['P'+str(camno)]
        P.append(np.dot(np.array([[scales[camno], 0.0, 0.0], [0.0, scales[camno], 0.0], [0.0, 0.0, 1.0]]), P_tmp))
        P_inv.append(np.linalg.pinv(P[-1]))
        vids.append(cv2.VideoCapture(os.path.join(args.drive_path, camid+'.mp4')))
    d = scipy.io.loadmat(args.calib_path)['d']
    K = scipy.io.loadmat(args.calib_path)['K']

    # load camera timestamps
    with open(os.path.join(args.drive_path, 'camera_vsync_output.txt')) as f:
        TS = f.readlines()

    # create necessary output directories and files
    if not os.path.exists('/home/akshay/.tmp'):
        os.makedirs('/home/akshay/.tmp')
    out = h5py.File(os.path.join('/home/akshay/.tmp', 'gpp.h5'), 'w')

    
    frame_count = 0
    frame_total = len(TS)

    while (vids[0].isOpened()):
        for camno in range(len(camera_ids)):
            # load image
            ret, raw_image = vids[camno].read()
            raw_image = cv2.undistort(raw_image, K, d)
            if camno == 0 or camno == 3:
            	raw_image = raw_image[:1000, :, :]

            # preprocess image for network
            image = preprocess_image(raw_image)
            image, scale = resize_image(image)

            # construct inputs
            inputs = [np.expand_dims(image, axis=0), np.expand_dims(P_inv[camno], axis=0), np.expand_dims(plane_params[camno], axis=0)]

            # process image
            start = time.time()
            # run network
            boxes, dimensions, scores, labels, orientations, keypoints, keyplanes, residuals = model.predict_on_batch(inputs)[:8]

            # correct for image scale
            boxes /= scale

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
                x_all = P[camno] @ X_all
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
            dset = out.create_dataset('results/' + camera_ids[camno] + '/boxes/' + TS[frame_count][:13], boxes[:, :4].shape, dtype='f2')
            dset[...] = boxes[:, :4]
            dset = out.create_dataset('results/' + camera_ids[camno] + '/keypoints/' + TS[frame_count][:13], boxes[:, 4:].shape, dtype='f2')
            dset[...] = boxes[:, 4:]
            dset = out.create_dataset('results/' + camera_ids[camno] + '/labels/' + TS[frame_count][:13], labels.shape, dtype='u1')
            dset[...] = labels
            dset = out.create_dataset('results/' + camera_ids[camno] + '/scores/' + TS[frame_count][:13], scores.shape, dtype='f2')
            dset[...] = scores
            dset = out.create_dataset('results/' + camera_ids[camno] + '/locations/' + TS[frame_count][:13], locations.shape, dtype='f4')
            dset[...] = locations
            dset = out.create_dataset('results/' + camera_ids[camno] + '/angles/' + TS[frame_count][:13], angles.shape, dtype='f4')
            dset[...] = angles
            dset = out.create_dataset('results/' + camera_ids[camno] + '/dimensions/' + TS[frame_count][:13], dimensions.shape, dtype='f4')
            dset[...] = dimensions
            dset = out.create_dataset('results/' + camera_ids[camno] + '/residuals/' + TS[frame_count][:13], residuals.shape, dtype='f2')
            dset[...] = residuals


        print("{} Cameras, Frame {}/{} at {:.2f}Hz".format(len(camera_ids), frame_count, frame_total, len(camera_ids)*1.0 / (time.time() - start)))
        frame_count += 1
    
    for camno in range(len(camera_ids)):
        vids[camno].release()
    out.close()

if __name__ == '__main__':
    main()