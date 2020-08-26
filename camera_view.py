import numpy as np
import cv2
import time

import NI_camera

while True:
    #start = time.time()
    display = NI_camera.capture()
    #stop = time.time()
    #print(stop - start)
    cv2.imshow('display',display/display.max())
    cv2.waitKey(1)
    print(display.max())