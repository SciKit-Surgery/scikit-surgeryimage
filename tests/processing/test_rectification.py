# coding=utf-8

import numpy as np
import cv2
from sksurgeryimage.utilities.utilities import are_similar

def test_rectify_4_1_1_26():
    """
    A unit test for the performance of OpenCV's rectification test
    function. This isn't really part of scikit-surgeryimage, but I've added
    it as a convenience. We got into some trouble when we changed versions
    of opencv-contrib-python from 4_1_0_25 to 4_1_0_26. OpenCV's underlying
    rectification function doesn't seem too stable between versions, so this unit test
    should pick up changes due to changes in opencv-contrib-python.
    As of Issue #97 we're using sksurgeryimage.utilities.are_similar for testing
    which gives us a bit more leeway on image similarity.
    """
    original = np.zeros((64,128,3), dtype=np.uint8)

    for row in range(8):
        for col in range(128):
            for channel in range(3):
                original[row*8,col,channel] = 255

    for col in range(16):
        for row in range(64):
            for channel in range(3):
                original[row,col*8,channel] = 255

    li = np.eye(3,3,dtype = np.double)
    li[0,0] = 120.120
    li[0,2] = 32.0
    li[1,1] = 147.147
    li[1,2] = 32.0
    ld = np.zeros((1,4),dtype = np.double)
    ld = (-0.333, 0.088, 0.022, 0.004)

    rectify_rotation = [None, None]
    rectify_projection = [None, None]
    rectify_valid_roi = [None, None]
    rectify_dx = [None, None]
    rectify_dy = [None, None]
    #this looks the wrong way round
    image_size = (original.shape[1], original.shape[0])
    rectify_new_size = image_size
    sr = np.array([[9.9998654509984819e-01, -4.0289369251237809e-03, -3.2676117460306965e-03] ,
         [4.0291930116066648e-03, 9.9999188019182839e-01, 7.1791971146594814e-05] ,
         [3.2672959683266610e-03, -8.4956843604492188e-05, 9.9999465876543070e-01 ]])
    st = np.zeros((3,1), dtype=np.double)
    st[0,0]=1.1
    rectify_rotation[0], \
    rectify_rotation[1], \
    rectify_projection[0], \
    rectify_projection[1], \
    rectify_q, \
    rectify_valid_roi[0], \
    rectify_valid_roi[1] = \
    cv2.stereoRectify(li,
                      ld,
                      li,
                      ld,
                      image_size,
                      sr,
                      st,
                      flags=cv2.CALIB_ZERO_DISPARITY,
                      alpha=0.0,
                      newImageSize=rectify_new_size
                      )


    for image_index in range(2):
        rectify_dx[image_index], rectify_dy[image_index] = cv2.initUndistortRectifyMap(
                        li,
                        ld,
                        rectify_rotation[image_index],
                        rectify_projection[image_index],
                        rectify_new_size,
                        cv2.CV_32FC1
                        )


    rectified_image = cv2.remap(original,
                                rectify_dx[0],
                                rectify_dy[0],
                                cv2.INTER_LINEAR
                               )

    expected_rectified = cv2.imread('tests/data/processing/rectified_image_left_4.1.1.26.png')

    assert are_similar(rectified_image, expected_rectified,
            threshold = 0.800, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.006)

    rectified_image = cv2.remap(original,
                                rectify_dx[1],
                                rectify_dy[1],
                                cv2.INTER_LINEAR
                               )

    expected_rectified = cv2.imread('tests/data/processing/rectified_image_right_4.1.1.26.png')

    assert are_similar(rectified_image, expected_rectified,
            threshold = 0.800, metric = cv2.TM_CCOEFF_NORMED,
            mean_threshold = 0.005)


