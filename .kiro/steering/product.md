# Product Overview

scikit-surgeryimage is a Python library providing image processing algorithms for image-guided surgery. It is part of the SciKit-Surgery ecosystem, developed at the Wellcome EPSRC Centre for Interventional and Surgical Sciences at University College London (UCL).

## Key Capabilities

- Video acquisition: convenience classes for camera capture and video read/write (mono and stereo)
- Camera calibration: point detection using chessboards, ArUco markers, ChArUco boards, and dotty grids
- Image processing: interlacing/deinterlacing, morphological operators (erosion, dilation)
- Utilities: camera enumeration, text overlay, WEISS logo rendering

## Domain Context

This is a medical/scientific imaging library. Users are researchers and developers building computer-assisted surgery systems. Accuracy of point detection and calibration is critical. The library wraps and extends OpenCV functionality with surgery-specific workflows.

## License

BSD-3-Clause. Copyright 2018 University College London.
