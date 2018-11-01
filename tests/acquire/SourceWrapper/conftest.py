import cv2
import numpy as np 
import pytest
from sksurgeryimage.acquire import SourceWrapper

@pytest.fixture(scope="function")
def source_wrapper():
    return SourceWrapper.VideoSourceWrapper()