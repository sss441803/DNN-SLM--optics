import os
from library import *
# This allows libaries to load without using GPU if desired
if use_gpu == False:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from numba import cuda
if cuda.is_available() and use_gpu == True:
    gpu_available = True
    from cuda_init import *
    print('GPU used')
else:
    gpu_available = False
    from init import *
    print('GPU not used')

from tensorflow import keras # Machine learning library
import numpy as np # Number array library
import matplotlib.pyplot as plt # Library for plotting
import cv2 # Library for displaying windows and rescaling images
import time
import tkinter as tk # Library for slider control bar
from cv2 import VideoWriter, VideoWriter_fourcc # Libraries for recording videos from camera captured images

import NI_camera

# Loading pretrained model
model = keras.models.load_model(model_dir, compile = False)

running = True  # Global flag of whether the control loop is running
idx = 0  # loop index
L = 0 # Lambda decay factor
P = 0 # Gain ('Proportional')
I = 0 # Integration term factor
D = 0 # Derivative term factor. Try to keep it zero as it does not really improve aberration correction in experience

def start():
    """Enable scanning of the slider control pannel by setting the global flag to True."""
    global running # Global flag within the definition of a function allows accessing a variable outside the function
    running = True

def stop():
    """Stop scanning by setting the global flag to False."""
    global running
    running = False

def param_update():
    """Update the parameter to the algorithm"""
    global L
    global P
    global I
    global D
    L = L_slider.get() # Acquires slider value
    P = P_slider.get()
    I = I_slider.get()
    D = D_slider.get()

# Setting up slider control panel layout
root = tk.Tk()
root.title("PID control")
root.geometry("250x600")
app = tk.Frame(root)
app.grid()
start = tk.Button(app, text="Start Scan", command=start)
stop = tk.Button(app, text="Stop", command=stop)
start.grid()
stop.grid()
L_slider = tk.Scale(app, label='L', orient = tk.HORIZONTAL, from_=0, to=1, resolution = 0.05)
P_slider = tk.Scale(app, label='P', orient = tk.HORIZONTAL, from_=0, to=1, resolution = 0.05)
I_slider = tk.Scale(app, label='I', orient = tk.HORIZONTAL, from_=0, to=1, resolution = 0.05)
D_slider = tk.Scale(app, label='D', orient = tk.HORIZONTAL, from_=0, to=1, resolution = 0.05)
update = tk.Button(app, text='Update', command=param_update)
L_slider.grid()
P_slider.grid()
I_slider.grid()
D_slider.grid()
update.grid()

"""This function should be applied to correct for physical aberration. Parameters of PID control will be asked.
Function can take a parameter 'total' as the total number of frames to run."""
def physical(total = 1000):
    global running
    global idx
    global L
    global P
    global I
    global D
    L = float(input('L'))
    P = float(input('P'))
    I = float(input('I'))
    D = float(input('D'))
    cv2.namedWindow("SLM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SLM", SLM_width, SLM_height)
    output = np.zeros(num_of_terms)
    error = np.zeros(num_of_terms)
    errors = np.zeros((5, num_of_terms))
    cumulative_error = np.zeros(num_of_terms)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = VideoWriter('physical turbulence.avi', fourcc, float(20), (cam_y, cam_x))
    for i in range(total):
        if i % 10 == 0:
            root.update()
        if running == True:
            display = NI_camera.capture()
            video_frame = 15 * np.sqrt(display).astype(np.int8).T
            video_frame = cv2.merge((video_frame, video_frame, video_frame))
            video.write(video_frame)
            cv2.waitKey(1)
            cv2.imshow('image', display / display.max())
            cv2.waitKey(1)
            image, x_center, y_center = image_crop(display[:, :cam_x//2])
            std_corrected = std_measure(image)
            print(std_corrected)
            image = image_disect(display)
            dim_image = image.reshape(1, input_x, input_y, 2)
            dim_image = dim_image.astype('float64')
            error = np.append((0, 0), - model(dim_image, training=False)[0] / scaling_factor)
            errors = np.roll(errors, 1, 0)
            errors[0] = error
            derror = errors[0] - errors[2]
            '''errors[2] means you find the difference between the current estimated
            errors and the errors two frames ago. This is for the derivative term. If D is zero, this does not make
            any difference'''
            output = P * (error + I * cumulative_error + D * derror)
            cumulative_error = cumulative_error * L + error
            h_result = add(output)
            cv2.moveWindow("SLM", SLM_x, SLM_y)
            cv2.imshow("SLM", h_result)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    video.release()

"""This function does aberration correction on simulated turbulence.
Total frames, speed of update and fried parameter can be entered.
Speed ranges from 0 to 1 (static to no correlation between frames).
Fried parameter determines the strength of aberration."""
def simulated(total=100, speed=0.05, fried=5):
    global running
    global idx
    global L
    global P
    global I
    global D
    cv2.namedWindow("SLM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SLM", SLM_width, SLM_height)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = VideoWriter('Simulated.avi', fourcc, float(20), (cam_y, cam_x))
    output = np.zeros(num_of_terms)
    error = np.zeros(num_of_terms)
    previous_error = np.zeros(num_of_terms)
    cumulative_error = np.zeros(num_of_terms)
    B = np.random.normal(0, np.sqrt(S.diagonal()))
    for i in range(total):
        if idx % 10 == 0:
            root.update()
        if running == True:
            cv2.moveWindow("SLM", SLM_x, SLM_y)
            cv2.imshow("SLM", np.zeros((SLM_width, SLM_height)))
            cv2.waitKey(1)
            time.sleep(0.2)
            perfect = NI_camera.capture()
            std_perfect = std_measure(image_crop(perfect[:, :cam_x//2])[0])
            B = B * np.sqrt(1-speed) + np.random.normal(0, np.sqrt(S.diagonal())) * np.sqrt(speed)
            A = X.dot(B)
            aberration = index_matrix.dot(A) * np.power(fried, 5 / 6)
            aberration[0] = 0
            aberration[1] = 0
            h_result = add(aberration)
            cv2.moveWindow("SLM", SLM_x, SLM_y)
            cv2.imshow("SLM", h_result)
            cv2.waitKey(1)
            time.sleep(0.2)
            uncorrected = NI_camera.capture()
            std_uncorrected = std_measure(image_crop(uncorrected[:, :cam_x//2])[0])
            cv2.imshow('uncorrected', uncorrected / uncorrected.max())
            cv2.waitKey(1)
            h_result = add(aberration + output)
            cv2.moveWindow("SLM", SLM_x, SLM_y)
            cv2.imshow("SLM", h_result)
            cv2.waitKey(1)
            time.sleep(0.2)
            corrected = NI_camera.capture()
            std_corrected = std_measure(image_crop(corrected[:, :cam_x//2])[0])
            print(std_uncorrected, std_corrected, std_perfect)
            cv2.imshow('corrected', corrected / corrected.max())
            cv2.waitKey(1)
            image, x_center, y_center = image_crop(corrected)
            video_frame = 15*np.sqrt((np.concatenate((uncorrected[:, :cam_x//2], corrected[:, :cam_x//2])))).astype(np.int8).T
            video_frame = cv2.merge((video_frame, video_frame, video_frame))
            video.write(video_frame)
            cv2.waitKey(1)
            image = image_disect(corrected)
            dim_image = image.reshape(1, input_x, input_y, 2)
            dim_image = dim_image.astype('float64')
            error =  np.append((0, 0), - model(dim_image, training=False)[0] / scaling_factor)
            np.where(cumulative_error > 2, 2, cumulative_error)
            derror = error - previous_error
            output = P * (error + I * cumulative_error - D * derror)
            cumulative_error = cumulative_error * L + error
    cv2.destroyAllWindows()
    video.release()

def predict_w_local_files(beginning=0, end=20):
    for i in range(beginning, end):
        image_name = image_dir + '\image_{}.npy'.format(i)
        image = np.load(image_name)
        cv2.imshow('image', image/image.max())
        cv2.waitKey(1)
        coeff = np.load(label_dir + '\coefficient_{}.npy'.format(i)) * scaling_factor
        dim_image = image_disect(image)
        dim_image = dim_image.reshape(1, input_x, input_y, 2)
        dim_image = dim_image.astype('float64')
        prediction = model.predict(dim_image)
        diff = abs(coeff[2 :num_of_terms] - prediction[0])
        plt.plot(np.array((abs(coeff[2 :num_of_terms]), abs(coeff[2 :num_of_terms] - prediction[0]))).transpose())
        plt.legend(['actual', 'difference'])
        plt.show()
        print((coeff[2 :num_of_terms]**2).sum(), (diff**2).sum())

def predict_w_generated(total=20, max=4):
    cv2.namedWindow("SLM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SLM", SLM_width, SLM_height)
    for i in range(total):
        coefficient = coeff_gen(np.random.uniform(0, max))
        h_result = add(coefficient)
        cv2.moveWindow("SLM", SLM_x, SLM_y)
        cv2.imshow("SLM", h_result)
        cv2.waitKey(1)
        time.sleep(0.2)
        display = NI_camera.capture()
        cv2.imshow('image', display / display.max())
        cv2.waitKey(1)
        image = image_disect(display)
        dim_image = image.reshape(1, input_x, input_y, 2)
        dim_image = dim_image.astype('float64')
        prediction = model.predict(dim_image)
        coeff_to_slm = np.append((0, 0), coefficient[2: num_of_terms] - prediction[0] / scaling_factor)
        h_result = add(coeff_to_slm)
        cv2.moveWindow("SLM", SLM_x, SLM_y)
        cv2.imshow("SLM", h_result)
        cv2.waitKey(1)
        time.sleep(0.2)
        display_corrected = NI_camera.capture()
        cv2.imshow('corrected', display_corrected / display_corrected.max())
        cv2.waitKey(1)
        diff = abs(coefficient[2 :num_of_terms] - prediction[0] / scaling_factor)
        print((coefficient[2 :num_of_terms] ** 2).sum(), (diff ** 2).sum())
        time.sleep(1)
        plt.plot(np.array((abs(coefficient[2 :num_of_terms]), abs(diff))).transpose())
        plt.legend(['actual', 'difference'])
        plt.show()

#predict_w_local_files()
#predict_w_generated()
#physical()
#simulated()