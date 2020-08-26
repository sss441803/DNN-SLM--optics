In stall the following python libraries: tensorflow, matplotlib, numpy, cv2, numba, pyserial, scipy, IPython, nivision, ctypes.

If you want to use GPU acceleration, make sure you have an Nvidia CUDA capable GPU. Install CUDA 10.1 to avoid version conflict.

If you want to use PyCharm IDE for the project, the project folder should be copied to the venv folder under the project folder. Otherwise, directly running the scripts and editing them in an augmented text editor such as visual studio code is fine. Running the code in the virtual environment of PyCharm might be problematic. If so, try running from the command prompt directly.

To run a script or access functions in a script, you need to have the CMD in the correct directory. Copy the directory of the project folder where the scripts are. Open CMD. Input 'cd: project folder' and enter the line. 'cd' stands for change directory. You should see the desired directory shown before where you enter new lines of code. Sometimes, if the disk of the directory is not the default directory (e.g. you start with default directory in disk C and the project is in D), then you have to type a new line of code 'd:' and enter it to change the directory successfully.

If you want to run any script, type 'python script_name.py'. If you want to run a function in the script, type 'python'. You should enter interactive mode of python console, and there you can type 'import library_name' or 'from library name import function_name'. Here, the library_name is the name of the script .py file containing the function without the .py extension. For example, to run function image_crop in the script library.py, run 'from library import image_crop'.

library.py contains all the parameters needed to adjust, such as camera active area, SLM window size, neural network input size, etc.

First, setup your optics. To access the camera view, run camera_view.py.

Second, acquire training images. Make sure the SLM is operating in phase modulation mode. Specify the total number of image used for training and validation in library.py. Run the script image_collect.py. You should start with a small number of images to acquire, train the network and evaluate the network with predictor.py (discussed later) just to see there are no bugs. Beware, training of the nerual network requires patience. Sample size needs to be very large (100,000 images should be good).

Third, train your neural network. Run Net.py.

Forth, evaluate the performance of the network using predictor.py. You can import all functions in prector by 'from predictor import *'. Run the four functions available in the library to evaluate performance. You want to see the script itself to understand what each does.