# coding=utf-8


def identify_cameras():
    """
    Show images from each camera, with the camera input number overlaid.
    """
    num_cameras = count_cameras()
    cameras = []

    for i in range(num_cameras):
        cam = cv2.VideoCapture(i)
        cameras.append(cam)

    while True:
        for i in range(num_cameras):
            ret, frame = cameras[i].read()
            title_string = "Camera: " + str(i) + ". Press q to quit."

            if ret:
                text_overlay_properties = prepare_cv2_text_overlay(i, frame)
                cv2.putText(frame, *text_overlay_properties)

                cv2.imshow(title_string, frame)

            else:
                logging.info("Camera %s not grabbing", i)

        if cv2.waitKey(1) == ord('q'):
            break

    for i in range(num_cameras):
        cameras[i].release()

