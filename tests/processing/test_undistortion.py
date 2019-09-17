# coding=utf-8

import numpy as np
import cv2


def test_undistort_4_1_0_25():
    """
    A unit test for the performance of OpenCV's undistort
    function. This isn't really part of scikit-surgeryimage, but I've added
    it as a convenience. We got into some trouble when we changed versions
    of opencv-contrib-python from 4_1_0_25 to 4_1_0_26. OpenCV's underlying
    undistort function is only tested to a tolerance of 16 for a uint8 image,
    so we do the same here.
    """
    original = np.zeros((64,64,3), dtype=np.uint8)
    absolute_tolerance = 16

    for row in range(8):
        for col in range(64):
            for channel in range(3):
                original[row*8,col,channel] = 255

    for col in range(8):
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
    undistorted = cv2.undistort ( original, li, ld )

    expected_undistorted = cv2.imread('tests/data/processing/undistorted_4.1.0.25.png')

    np.testing.assert_allclose(undistorted, expected_undistorted, rtol = 0.0, atol = absolute_tolerance)

