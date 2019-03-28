#coding=utf-8
""" Class to crop an image. """

import cv2
import numpy as np

class ImageCropper():
    """ Class to crop an image. """
    def __init__(self):

        self.img = None
        self.img_with_rect = None
        self.window_name = None
        self.done = False
        self.selecting = False
        self.start_x = 0
        self.end_x = 0
        self.start_y = 0
        self.end_y = 0
        self.roi = []


    def crop(self, img):
        """ Crop an image by selecting a rectaungular region with the mouse.
        :param img: input image.
        :type img: numpy array
        :return: roi - length 2 array of tuples,
                        [ (start_x, start_y), (end_x, end_y)]
        """
        self.img = np.copy(img)
        self.window_name = "Crop Image"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.click_to_crop)
        cv2.imshow(self.window_name, self.img)

        while not self.done:
            cv2.waitKey(1)

        cv2.destroyWindow(self.window_name)

        self.check_start_and_end_not_equal()
        self.check_order_of_start_end_points()

        self.roi = []
        start_xy = (self.start_x, self.start_y)
        end_xy = (self.end_x, self.end_y)
        self.roi.append(start_xy)
        self.roi.append(end_xy)

        return self.roi

    def click_to_crop(self, event, x, y, flags, param):
        #pylint:disable=unused-argument, invalid-name
        """ Callback to select the start/end points of roi.
        Left button down starts drawing, left button up stops drawing. """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x = x
            self.start_y = y
            self.selecting = True

        elif self.selecting and event == cv2.EVENT_MOUSEMOVE:
            self.img_with_rect = np.copy(self.img)
            cv2.rectangle(self.img_with_rect,
                          (self.start_x, self.start_y),
                          (x, y), (0, 255, 0))

            cv2.imshow(self.window_name, self.img_with_rect)
            cv2.waitKey(1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_x = x
            self.end_y = y

            self.done = True

    def check_start_and_end_not_equal(self):
        """ If the start and end coordinates are the same, set the ROI
        to the entire image (don't do any cropping). """

        if self.start_x == self.end_x or self.start_y == self.end_y:
            print("Cropping area has dimension 0, setting ROI to entire image")
            self.start_x = 0
            self.start_y = 0
            self.end_y, self.end_x = self.img.shape[:2]

    def check_order_of_start_end_points(self):
        """ The end x/y coordintates of the rectangle could be less
        than the start x/y. If so, swap them round, otherwise we can't use
        start_x:end_x as a slice to a numpy array. """

        if self.end_x < self.start_x:
            self.start_x, self.end_x = self.end_x, self.start_x

        if self.end_y < self.start_y:
            self.start_y, self.end_y = self.end_y, self.start_y
