# coding=utf-8

"""
Functions to check cameras.
"""
import logging
import cv2

LOGGER = logging.getLogger(__name__)


def count_cameras():
    """
    Count how many camera sources are available.
    This is done by trying to instantiate cameras 0..9,
    and presumes they are in order, sequential, starting
    from zero.
    :returns: int, number of cameras
    """
    max_cameras = 10

    found_cameras = 0
    for i in range(max_cameras):
        cam = cv2.VideoCapture(i)
        ret = cam.isOpened()
        cam.release()

        if not ret:
            found_cameras = i
            break

    logging.info("Found %d cameras", found_cameras)
    return found_cameras
