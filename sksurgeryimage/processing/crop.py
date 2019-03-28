
import cv2
import numpy as np

class ImageCropper():
    
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
        
        self.roi = []
        self.done = False
        self.selecting = False
        while not self.done:
            cv2.waitKey(1)
        
        cv2.destroyWindow(self.window_name)
        return self.roi

    def click_to_crop(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi = [(x, y)]
            self.selecting = True

        if self.selecting and event == cv2.EVENT_MOUSEMOVE:
                self.img_with_rect = np.copy(self.img)
                cv2.rectangle(self.img_with_rect, self.roi[0], (x,y), (0, 255, 0))
                cv2.imshow(self.window_name, self.img_with_rect)
                cv2.waitKey(1)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi.append((x, y))
            self.done = True
            
