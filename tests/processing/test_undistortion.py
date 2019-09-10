# coding=utf-8

import numpy as np
import cv2


def test_undistort():
    original = np.zeros((64,64,3),dtype = np.uint8)

    for row in range(8):
        for col in range(64):
            for channel in range (3):
                original[row*8,col,channel]=255

    for col in range(8):
        for row in range(64):
            for channel in range (3):
                original[row,col*8,channel]=255

    cv2.imwrite('tests/output/pattern.png', original)
    
    li = np.eye(3,3,dtype = np.double)
    li[0,0] = 120.120
    li[0,2] = 32.0
    li[1,1] = 147.147
    li[1,2] = 32.0
    ld = np.zeros((1,4),dtype = np.double)
    ld = (-0.333, 0.088, 0.022, 0.004)
    undistorted = cv2.undistort ( original, li, ld )
    cv2.imwrite('tests/output/undistorted.png', undistorted)

    expected_undistorted_4_1_0_25 = cv2.imread('tests/data/processing/undistorted_4.1.0.25.png')
    expected_undistorted_4_1_1_26 = cv2.imread('tests/data/processing/undistorted_4.1.1.26.png')

    np.testing.assert_array_equal(undistorted, expected_undistorted_4_1_0_25)


if __name__ == "__main__":
    
    test_undistort()
