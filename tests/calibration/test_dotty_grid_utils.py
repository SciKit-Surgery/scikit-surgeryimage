# -*- coding: utf-8 -*-

import os
import datetime
import logging
import numpy as np
import cv2 as cv2
from sksurgeryimage.calibration.dotty_grid_point_detector import DottyGridPointDetector


def __write_annotated_image(image, ids, image_points, file_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for counter in range(ids.shape[0]):
        cv2.putText(image, str(ids[counter][0]), (int(image_points[counter][0]), int(image_points[counter][1])), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    split_path = os.path.splitext(file_name)
    previous_dir = os.path.dirname(split_path[0])
    previous_dir = os.path.basename(previous_dir)
    base_name = os.path.basename(split_path[0])
    output_file = os.path.join('tests/output', base_name + '_' + previous_dir + '_labelled.png')
    cv2.imwrite(output_file, image)


def __check_real_image(model_points,
                       image_file_name,
                       intrinsics_file_name,
                       distortion_file_name,
                       is_metal=False
                       ):
    logging.basicConfig(level=logging.DEBUG)
    image = cv2.imread(image_file_name)
    intrinsics = np.loadtxt(intrinsics_file_name)
    distortion = np.loadtxt(distortion_file_name)

    size = (2600, 1900)
    fiducials = [133, 141, 308, 316]
    if is_metal:
        fiducials = [69, 74, 149, 154]
        size = (1360, 1200)

    detector = DottyGridPointDetector(model_points,
                                      fiducials,
                                      intrinsics,
                                      distortion,
                                      reference_image_size=size
                                      )

    time_before = datetime.datetime.now()

    ids, object_points, image_points = detector.get_points(image)

    time_after = datetime.datetime.now()
    time_diff = time_after - time_before

    print("__check_real_image:time_diff=" + str(time_diff))

    __write_annotated_image(image, ids, image_points, image_file_name)

    model = detector.get_model_points()
    assert model.shape[0] == model_points.shape[0]

    return ids.shape[0]  # , image_points

def __check_real_OR_image(model_points,
                         image_file_name,
                         intrinsics_file_name,
                         distortion_file_name,
                         is_metal=False
                         ):
    logging.basicConfig(level=logging.DEBUG)
    image = cv2.imread(image_file_name)
    intrinsics = np.loadtxt(intrinsics_file_name)
    distortion = np.loadtxt(distortion_file_name)

    size = (2600, 1900)
    fiducials = [133, 141, 308, 316]
    if is_metal:
        fiducials = [133, 141, 308, 316]
        size = (2600, 1900)

    detector = DottyGridPointDetector(model_points,
                                      fiducials,
                                      intrinsics,
                                      distortion,
                                      reference_image_size=size
                                      )

    time_before = datetime.datetime.now()

    ids, object_points, image_points = detector.get_points(image)

    time_after = datetime.datetime.now()
    time_diff = time_after - time_before

    print("__check_real_image:time_diff=" + str(time_diff))

    __write_annotated_image(image, ids, image_points, image_file_name)

    model = detector.get_model_points()
    assert model.shape[0] == model_points.shape[0]

    return ids.shape[0]  # , image_points

