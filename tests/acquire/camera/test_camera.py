import pytest
import cv2
import numpy as np
from sksurgeryimage.acquire import camera, utilities


def test_is_single_camera_input():

    single_camera = 1
    multi_cameras = [1,2,3]

    assert camera.CameraWrapper.is_single_camera_input(single_camera)
    assert not camera.CameraWrapper.is_single_camera_input(multi_cameras)


def test_if_camera_available():
    """ If there is a camera available, test some of the functions that need a camera input"""

    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        cam.release()

        run_tests_with_camera()

def run_tests_with_camera():
    """ Test functions that require a camera input.
    Don't preface function names with test_ to avoid autodiscovery, as this
    will run them even if no camera is available.
    """

    cam_wrapper = add_camera_test()
    check_camera_open_and_release_test(cam_wrapper)

def add_camera_test():
        """This runs add_cameras, add_single_cameras and update_output_video_dimensions. """

        print("In add_camera_test")

        cam_wrapper = camera.CameraWrapper()
        cam_wrapper.add_cameras(0)

        assert len(cam_wrapper.cameras) == 1
        assert len(cam_wrapper.frames) == 1

        cam = cam_wrapper.cameras[0]

        cam_width = int(cam.get(3))
        cam_height = int(cam.get(4))

        assert cam_wrapper.frames[0].shape == (cam_width, cam_height, 3)

        # Only one camera input, so the output should have the same dimensions
        assert cam_wrapper.output_array.shape[:2] == (cam_height, cam_width)

        return cam_wrapper


def check_camera_open_and_release_test(cam_wrapper):
        
        print("In check_camera_open_and_release_test")

        assert cam_wrapper.are_all_cameras_open()
        cam_wrapper.release_cameras()
        assert not cam_wrapper.are_all_cameras_open()


        




