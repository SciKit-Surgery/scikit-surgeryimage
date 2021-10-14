# coding=utf-8

"""
Various utilities, like preparing overlay text.
"""
import cv2
import sksurgerycore.utilities.validate as scv


def prepare_cv2_text_overlay(overlay_text, frame, text_scale=1):
    """
    Return settings for text overlay on a cv2 frame.
    """
    scv.validate_is_string_or_number(overlay_text)

    text = str(overlay_text)
    text_y_offset = 10
    text_location = (0, frame.shape[0] - text_y_offset)  # Bottom left
    text_colour = (255, 255, 255)

    text_overlay_properties = (
        text, text_location, cv2.FONT_HERSHEY_COMPLEX, text_scale, text_colour)

    return text_overlay_properties


def noisy_image(image, mean=0, stddev=(50, 5, 5)):
    """
    Creates a noise image, based on the dimensions of the
    passed image.
    param: the image to define size and channels of output
    returns: a noisy image
    """
    cv2.randn(image, (mean), (stddev))
    return image


def are_similar(image0, image1, threshold = 0.995,
        metric = cv2.TM_CCOEFF_NORMED,
        mean_threshold = 0.005):
    """
    Compares two images to see if they are similar.

    :param image0, image0: The images
    :param threshold: The numerical threshold to use, default 0.995
    :param method: The comparison metric, default normalised cross correlation,
        cv2.TM_CCOEFF_NORMED
    :param mean_threshold: Also compare the mean values of each array,
        return false if absolute difference of image means divided by the
        average of both images is greater than the mean_threshold, if less
        than zero this test will be skipped

    :return: True if the metric is greater than the threshols, false otherwise
        or if the images are not the same dimensions or type
    """

    if image0.shape != image1.shape:
        return False

    if image0.dtype != image1.dtype:
        return False

    if cv2.matchTemplate(image0, image1, metric)[0] < threshold:
        return False

    return image_means_are_similar(image0, image1, mean_threshold)


def image_means_are_similar(image0, image1, threshold = 0.005):
    """
    Compares two images to see if they have similar mean pixel values

    :param image0, image0: The images
    :param threshold: The mean value threshold to use.
        return false if absolute difference of image means divided by the
        average of both images is greater than the mean_threshold.

    :return: false if absolute difference of image means divided by the
        average of both images is greater than the mean_threshold, true
        otherwise or if threshold is less than zero.

    """

    if threshold < 0.0:
        return True

    if image0.mean() == image1.mean():
        return True

    abs_mean_diff = abs(image0.mean() - image1.mean())
    normalising_mean = abs(image0.mean())
    if normalising_mean == 0.0:
        normalising_mean = abs(image1.mean())

    if abs_mean_diff / normalising_mean > threshold:
        return False

    return True
