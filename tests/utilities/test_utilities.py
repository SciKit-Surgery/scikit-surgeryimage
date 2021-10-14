# coding=utf-8

"""
Tests for utilities.py
"""
import pytest
import numpy as np
import cv2 
from sksurgeryimage.utilities import utilities


def test_prepare_text_overlay():

    frame_dims = (100, 100, 3)
    frame = np.empty(frame_dims, dtype=np.uint8)

    text_to_overlay = 1
    text_overlay_properties = utilities.prepare_cv2_text_overlay(
        text_to_overlay, frame)
    expected = ("1", (0, frame_dims[0] - 10),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    assert text_overlay_properties == expected

    scale = 10
    large_text_overlay_properties = utilities.prepare_cv2_text_overlay(
        text_to_overlay, frame, scale)
    expected_large = (
        "1", (0, frame_dims[0] - 10), cv2.FONT_HERSHEY_COMPLEX, scale, (255, 255, 255))

    assert expected_large == large_text_overlay_properties


def test_noisy():
    """
    tests function to add noise to a colour image.
    """
    timage = np.zeros((10, 10, 1), np.uint8)
    image = utilities.noisy_image(timage)

    assert image.shape == (10, 10, 1) 

def test_are_simiar():
    """
    Tests for the image similarity function
    """
    #returns false if images are not the same size
    image0 = np.zeros((4, 5, 3), dtype = 'uint8')
    image1 = np.zeros((4, 5, 1), dtype = 'uint8')
    assert not utilities.are_similar(image0, image1)

    #returns false if images are not the same type
    image0 = np.zeros((4, 5, 3), dtype = 'uint8')
    image1 = np.zeros((4, 5, 3), dtype = 'uint16')
    assert not utilities.are_similar(image0, image1)

    #returns false if images are not the same
    image0 = np.zeros((4, 5, 3), dtype = 'uint8')
    image1 = np.ones((4, 5, 3), dtype = 'uint8')
    assert not utilities.are_similar(image0, image1)

    #returns true if images are not the same but are well correlated
    #and means not compared
    image0 = np.zeros((4, 5, 3), dtype = 'uint8')
    image1 = np.ones((4, 5, 3), dtype = 'uint8')
    assert utilities.are_similar(image0, image1, mean_threshold = 1.0)

    #returns false if images are not the same and not correlated
    image0 = np.ones((4, 5, 3), dtype = 'uint8')
    np.random.seed(0)
    image1 = np.random.randint(0, 10, size=(4, 5, 3), dtype = 'uint8')
    assert not utilities.are_similar(image0, image1, mean_threshold = 1.0)
    
    #returns true if both images are zeros
    image0 = np.zeros((4, 5, 3), dtype = 'uint8')
    image1 = np.zeros((4, 5, 3), dtype = 'uint8')
    assert utilities.are_similar(image0, image1, mean_threshold = 1.0)

    #returns true if images are the same
    image0 = np.ones((4, 5, 3), dtype = 'uint8')
    image1 = np.ones((4, 5, 3), dtype = 'uint8')
    assert utilities.are_similar(image0, image1)


def test_image_means_are_similar():
    """Tests for image_means_are_similar function"""
    #returns true if threshold less than zero
    image0 = np.zeros((4, 5, 3), dtype = 'uint8')
    image1 = np.ones((4, 5, 3), dtype = 'uint8')
    assert utilities.image_means_are_similar(image0, image1, threshold = -1.0)
    
    #returns true if images equal
    image0 = np.zeros((4, 5, 3), dtype = 'uint8')
    image1 = np.zeros((4, 5, 3), dtype = 'uint8')
    assert utilities.image_means_are_similar(image0, image1)

    #returns false if images unequal
    image0 = np.zeros((4, 5, 3), dtype = 'uint8')
    image1 = np.ones((4, 5, 3), dtype = 'uint8')
    assert not utilities.image_means_are_similar(image0, image1)
    assert not utilities.image_means_are_similar(image1, image0)
