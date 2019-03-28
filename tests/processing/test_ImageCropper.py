from sksurgeryimage.processing.crop import ImageCropper
import numpy as np
import pytest

@pytest.fixture
def img():
    img = np.ones((100, 100, 3), dtype=np.uint8)
    return img

def test_check_start_end_not_equal(img):

    cropper = ImageCropper()
    cropper.img = img

    cropper.start_x = 1
    cropper.start_y = 1

    cropper.end_x = 1
    cropper.end_y = 1

    cropper.check_start_and_end_not_equal()

    assert cropper.start_x == 0
    assert cropper.start_y == 0
    assert cropper.end_x == 100
    assert cropper.end_y == 100

def test_return_ok(img):

    cropper = ImageCropper()

    # Too fiddly to simulate mouse clicks, so set values as if
    # this has be done.
    # start_x > end_x ensures that
    # cropper.check_order_of_start_end_points() is executed in full
    cropper.start_x = 75
    cropper.end_x = 25
    
    cropper.start_y = 75
    cropper.end_y = 25
    cropper.done = True

    roi = cropper.crop(img)

    assert roi == [(25, 25), (75, 75)]
