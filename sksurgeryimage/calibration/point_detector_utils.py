# -*- coding: utf-8 -*-

""" Utilities, specific to the PointDetector stuff. """
import os
import copy
import cv2


def write_annotated_image(input_image, ids, image_points, image_file_name):
    """
    Takes an input image, copies it, annotates point IDs and writes
    to the testing output folder.
    """
    image = copy.deepcopy(input_image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for counter in range(ids.shape[0]):
        cv2.putText(image,
                    str(ids[counter][0]),
                    (int(image_points[counter][0]),
                     int(image_points[counter][1])),
                    font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    split_path = os.path.splitext(image_file_name)
    previous_dir = os.path.dirname(split_path[0])
    previous_dir = os.path.basename(previous_dir)
    base_name = os.path.basename(split_path[0])
    output_file = os.path.join('tests/output', base_name
                               + '_'
                               + previous_dir
                               + '_labelled.png')
    cv2.imwrite(output_file, image)
