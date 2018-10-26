import pytest
import cv2
import numpy as np
from sksurgeryimage.acquire import camera, utilities


def test_is_single_camera_input():

    single_camera = 1
    multi_cameras = [1,2,3]

    assert camera.CameraWrapper.is_single_camera_input(single_camera)
    assert not camera.CameraWrapper.is_single_camera_input(multi_cameras)


def test_add_camera():
    # If there is a camera available, run some tests
    cam = cv2.VideoCapture(0)
    if cam.isOpened():

        # Get the camera image dimensions for use in tests
        cam_width = int(cam.get(3))
        cam_height = int(cam.get(4))
        cam.release()

        cam_wrapper = camera.CameraWrapper()
        cam_wrapper.add_camera(0)

        assert len(cam_wrapper.cameras) == 1
        assert len(cam_wrapper.frames) == 1

        assert cam_wrapper.frames[0].shape == (cam_width, cam_height, 3)

        # Only one camera input, so the output should have the same dimensions
        assert cam_wrapper.output_array.shape[:2] == (cam_height, cam_width)




