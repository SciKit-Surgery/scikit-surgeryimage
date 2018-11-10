# coding=utf-8

"""
Tests for camera utilities.
"""

from sksurgeryimage.utilities import camera_utilities as cu


def test_count_cameras():
    # Difficult to write a unit test as can't know how many cameras to expect.
    # For now, make sure no exceptions are thrown
    cu.count_cameras()
