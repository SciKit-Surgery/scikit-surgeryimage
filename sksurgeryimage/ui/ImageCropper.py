#coding=utf-8
""" Class to crop an image. """

import logging
import cv2
import numpy as np

class ImageCropper():
    """ Class to crop an image.
    Example usage using Opencv to capture/display image:

        cam = cv2.VideoCapture(0)
        ret, img = cam.read()

        cropper = ImageCropper()
        roi = cropper.crop(img)

        start_x, start_y = roi[0]
        end_x, end_y = roi[1]

        cv2.imshow('Cropped image', img[start_y:end_y,
                                        start_x:end_x,
                                        :])

        cv2.waitKey(1000) # Display for 1 second

        """
    def __init__(self):

        self.img = None
        self.img_with_rect = None
        self.window_name = None
        self.done = False
        self.selecting = False
        self.roi = []

    def crop(self, img):
        """ Crop an image by selecting a rectaungular region with the mouse.
        :param img: input image.
        :type img: numpy array
        :return: roi - If valid roi selected, return array of tuples,
                        [(start_x, start_y), (end_x, end_y)]
                       Otherwise (invalid ROI selected, or operation aborted),
                       return None
        """
        self.img = np.copy(img)
        self.window_name = "Crop Image (Press a to abort/reset crop area)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_click_callback)
        cv2.imshow(self.window_name, self.img)

        while not self.done:
            key = cv2.waitKey(1)
            if key == ord('a'): # Abort key pressed
                self.roi = []
                self.done = True

        cv2.destroyWindow(self.window_name)

        if self.done and self.roi:
            self.validate_roi()

        return self.roi

    def mouse_click_callback(self, event, x, y, flags, param):
        #pylint:disable=unused-argument, invalid-name
        """ Callback to select the start/end points of roi.
        Left button down starts drawing, left button up stops drawing. """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi.append((x, y))
            self.selecting = True

        elif self.selecting and event == cv2.EVENT_MOUSEMOVE:
            self.img_with_rect = np.copy(self.img)
            cv2.rectangle(self.img_with_rect,
                          (self.roi[0]), (x, y), (0, 255, 0))

            cv2.imshow(self.window_name, self.img_with_rect)
            cv2.waitKey(1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.roi.append((x, y))
            self.done = True

    def validate_roi(self):
        """ Check that a valid roi has been selected:
        1. Must have dimensions > 0, otherwise set roi to [].
        2. Order the x/y point in asecnding order. e.g. if the second point
        has x/y coorindates that are less than the first point, swap them. """

        start_x, start_y = self.roi[0]
        end_x, end_y = self.roi[1]

        # Check that dimensions are > 0
        if start_x == end_x or start_y == end_y:
            logging.info("Cropping area has dimension 0, cannot set ROI.")
            self.roi = []
            return

        # Check that end_x/y > start_x/y
        if end_x < start_x:
            start_x, end_x = end_x, start_x

        if end_y < start_y:
            start_y, end_y = end_y, start_y

        self.roi[0] = (start_x, start_y)
        self.roi[1] = (end_x, end_y)
