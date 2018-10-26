# coding=utf-8

""" Functions to identify camera sources """
import logging
import cv2

LOGGER = logging.getLogger(__name__)

def count_cameras():
    """
    Count how many camera sources are available. 
    """
    max_cameras = 10

    for i in range(max_cameras):
        cam = cv2.VideoCapture(i)
        ret = cam.isOpened()
        cam.release()

        if not ret:
            break

    logging.info("Found %d cameras", i)
    return i


def identify_cameras():
    """
    Show images from each camera, with the camera input number overlaid.
    """
    num_cameras = count_cameras()
    cameras = []

    for i in range(num_cameras):
        cam = cv2.VideoCapture(i)
        cameras.append(cam)

    while(True):
        for i in range(num_cameras):
            ret, frame = cameras[i].read()
            title_string = "Camera: " + str(i) + ". Press q to quit."

            if ret:
                text_overlay_properties = prepare_cv2_text_overlay(i, frame)
                cv2.putText(frame, *text_overlay_properties)

                cv2.imshow(title_string,frame)

            else:
                logging.info("Camera %s not grabbing", i)
                
                
        if cv2.waitKey(1) == ord('q'):
            break
    
    for i in range(num_cameras):
        cameras[i].release()


def prepare_cv2_text_overlay(overlay_text, frame, text_scale = 1):
    """
    Return settings for text overlay on a cv2 frame
    """
    
    validate_text_input(overlay_text)

    text = str(overlay_text)
    text_y_offset = 10
    text_location = (0 , frame.shape[0] - text_y_offset) #Bottom left
    text_colour = (255, 255, 255)

    text_overlay_properties = (text, text_location, cv2.FONT_HERSHEY_COMPLEX, text_scale, text_colour)

    return text_overlay_properties

def validate_text_input(overlay_text):
    """
    Return an error if input isn't a string or number.
    """
    if not is_string_or_number(overlay_text):
        raise TypeError('Text overlay must be string or numeric')

def is_string_or_number(var):
    """
    Return true if the input variable is either a string or a numeric type.
    Return false otherwise.s
    """
    valid_types = (str, int, float)
    if isinstance(var, valid_types):
        return True

    return False