from sksurgeryimage.ui.ImageCropper import ImageCropper
import numpy as np
import pytest
import os

pytest.skip("Can't run these tests without a display")

@pytest.fixture
def img():
    img = np.ones((100, 100, 3), dtype=np.uint8)
    return img

def test_check_start_end_not_equal(img):

    cropper = ImageCropper()
    roi_same_values = [(1, 1), (1, 1)]
    # Too fiddly to simulate mouse clicks, so set roi and 
    # done variables as if an roi has been selected.
    cropper.roi = roi_same_values
    cropper.done = True
    roi = cropper.crop(img)

    assert roi == []
    
def test_reorder_start_end_points(img):

    cropper = ImageCropper()

    roi_wrong_order = [(75, 75), (25, 25)]
    roi_right_order = [(25, 25), (75, 75)]
    # Too fiddly to simulate mouse clicks, so set roi and 
    # done variables as if an roi has been selected.
    cropper.roi = roi_wrong_order
    cropper.done = True

    roi = cropper.crop(img)

    assert roi == roi_right_order

def test_valid_roi(img):

    cropper = ImageCropper()

    roi_right_order = [(25, 25), (75, 75)]
    # Too fiddly to simulate mouse clicks, so set roi and 
    # done variables as if an roi has been selected.
    cropper.roi = roi_right_order
    cropper.done = True

    roi = cropper.crop(img)

    assert roi == roi_right_order