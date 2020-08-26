import numpy as np
import cv2
import os
import time

from library import use_gpu, train_total, validation_total, input_x, input_y, label_dir, image_dir, map_x, map_y, SLM_width, SLM_height, SLM_x, SLM_y, image_disect, coeff_gen
import NI_camera

from numba import cuda
if cuda.is_available() and use_gpu == True:
    gpu_available = True
    from cuda_init import *
    print('GPU used')
else:
    gpu_available = False
    from init import *
    print('GPU not used')

def collect_images(total):
    cv2.namedWindow("SLM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SLM", SLM_width, SLM_height)
    for i in range(total):
        coeff = coeff_gen(np.random.uniform(0,4))
        np.save(label_dir + '\coefficient_{}.npy'.format(i), coeff)
        h_result = add(coeff)
        cv2.moveWindow("SLM", SLM_x, SLM_y)
        cv2.imshow("SLM", h_result)
        cv2.waitKey(1)
        time.sleep(0.15)
        display = NI_camera.capture()
        im = image_disect(display)
        cv2.imshow('in focus', np.reshape(im[:,:,0], (input_x, input_y)))
        cv2.imshow('out of focus', np.reshape(im[:,:,1], (input_x, input_y)))
        cv2.waitKey(1)
        np.save(image_dir + '\image_{}.npy'.format(i), display)

collect_images(train_total + validation_total)