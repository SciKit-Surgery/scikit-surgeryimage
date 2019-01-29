# coding=utf-8

"""
Functions to support morphological operators.

In many cases, these will just be convenience wrappers around OpenCV functions.
"""

import cv2


def erode_with_cross(src, dst=None, size=3, iterations=1):
    """
    Erodes an image with a cross element.
    OpenCV supports both grey scale and RGB erosion.

    :param src: source image
    :param dst: if provided, an image of the same size as src
    :param size: size of structuring element
    :param iterations: number of iterations
    :return: the eroded image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    eroded = cv2.erode(src, dst=dst, kernel=kernel, iterations=iterations)
    return eroded


def dilate_with_cross(src, dst=None, size=3, iterations=1):
    """
    Dilates an image with a cross element.
    OpenCV supports both grey scale and RGB erosion.

    :param src: source image
    :param dst: if provided, an image of the same size as src
    :param size: size of structuring element
    :param iterations: number of iterations
    :return: the eroded image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    dilated = cv2.dilate(src, dst=dst, kernel=kernel, iterations=iterations)
    return dilated
