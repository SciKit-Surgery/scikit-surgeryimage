# -*- coding: utf-8 -*-

""" Utilities, specific to the PointDetector stuff. """
import os
import copy
import cv2
import numpy as np


def get_annotated_image(input_image, ids, image_points, color=(0, 255, 0)):
    """
    Takes an input image, copies it, annotates point IDs and returns
    the annotated image.
    """
    image = copy.deepcopy(input_image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for counter in range(ids.shape[0]):
        cv2.putText(image,
                    str(ids[counter][0]),
                    (int(image_points[counter][0]),
                     int(image_points[counter][1])),
                    font, fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)
    return image


def write_annotated_image(input_image, ids, image_points, image_file_name):
    """
    Takes an input image, copies it, annotates point IDs and writes
    to the testing output folder.
    """
    annotated_image = get_annotated_image(input_image, ids, image_points)
    split_path = os.path.splitext(image_file_name)
    previous_dir = os.path.dirname(split_path[0])
    previous_dir = os.path.basename(previous_dir)
    base_name = os.path.basename(split_path[0])
    output_file = os.path.join('tests/output', base_name
                               + '_'
                               + previous_dir
                               + '_labelled.png')
    cv2.imwrite(output_file, annotated_image)


def get_intersect(a_1, a_2, b_1, b_2):
    """
    Returns the point of intersection of the lines
    passing through a2,a1 and b2,b1.

    See https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    :param a_1: [x, y] a point on the first line
    :param a_2: [x, y] another point on the first line
    :param b_1: [x, y] a point on the second line
    :param b_2: [x, y] another point on the second line
    """
    stacked = np.vstack([a_1, a_2, b_1, b_2])
    homogenous = np.hstack((stacked, np.ones((4, 1))))
    line_1 = np.cross(homogenous[0], homogenous[1])
    line_2 = np.cross(homogenous[2], homogenous[3])
    p_x, p_y, p_z = np.cross(line_1, line_2)
    if p_z == 0:  # lines are parallel
        return float('inf'), float('inf')
    return p_x/p_z, p_y/p_z


def get_number_of_points(ids: np.ndarray,
                         model: dict
                         ):
    """
    Counts how many ids in ids (Nx1 ndarray), are in the model (dict of {id: 3D point},
    """
    number_of_points = 0
    for i in range(ids.shape[0]):
        if ids[i][0] in model:
            number_of_points += 1
    return number_of_points
