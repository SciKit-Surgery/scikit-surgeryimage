import pytest
import numpy as np
from sksurgeryimage.acquire import camera



def test_one_frame():
    
    dims = [(640, 480)]

    horz_stack = camera.CameraWrapper.calculate_stacked_frame_dims(dims, "h")
    vert_stack = camera.CameraWrapper.calculate_stacked_frame_dims(dims, "v")

    assert horz_stack == vert_stack == dims[0]
        

def test_two_equal_frames():
    
    frameA = (640, 480)
    frameB = (640, 480)
    dims = [frameA, frameB]

    horz_stack = camera.CameraWrapper.calculate_stacked_frame_dims(dims, "h")
    vert_stack = camera.CameraWrapper.calculate_stacked_frame_dims(dims, "v")

    horz_expected = (640 + 640, 480)
    vert_expected = (640, 480 + 480)

    assert horz_expected == horz_stack
    assert vert_expected == vert_stack

def test_two_unequal_frames():
    
    frameA = (640, 480)
    frameB = (500, 500)
    dims = [frameA, frameB]

    horz_stack = camera.CameraWrapper.calculate_stacked_frame_dims(dims, "h")
    vert_stack = camera.CameraWrapper.calculate_stacked_frame_dims(dims, "v")

    horz_expected = (640 + 500, 500)
    vert_expected = (640, 480 + 500)

    assert horz_expected == horz_stack
    assert vert_expected == vert_stack

def test_three_frames():
    
    frameA = (640, 480)
    frameB = (500, 500)
    frameC = (1920, 1080)
    dims = [frameA, frameB, frameC]

    horz_stack = camera.CameraWrapper.calculate_stacked_frame_dims(dims, "h")
    vert_stack = camera.CameraWrapper.calculate_stacked_frame_dims(dims, "v")

    horz_expected = (640 + 500 + 1920, 1080)
    vert_expected = (1920, 480 + 500 + 1080)

    assert horz_expected == horz_stack
    assert vert_expected == vert_stack


def test_invalid_stack_direction():

    dims = (1,1)
    
    with pytest.raises(ValueError):
        camera.CameraWrapper.calculate_stacked_frame_dims(dims, "invalid_direction")






