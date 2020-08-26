import numpy as np
from scipy import ndimage
import cv2

'''GPU parameters'''
use_gpu = True
#defining number of threads in the block
blockSize = 256

'''SLM display window parameters'''
SLM_width, SLM_height = 1200, 1200 # This is the size of the window to display the phase profile
SLM_x, SLM_y = 1726, -50 # Offset of window displayed for SLM phase profile

'''Camera parameters'''
cam_x, cam_y, cam_offset_x, cam_offset_y = 448, 224, 0, 128
in_focus_normalization, out_of_focus_normalization = 350, 500

'''Parameters for Zernike polynomial maps used'''
#defining n,m index to sequential ordering conversion function
def nmToterm(n, m):
    if (n == 0):
        return 0
    else:
        return (m + n) // 2 + 1 + nmToterm(n - 1, n - 1)
# Size of Zernike polynomial maps. Can be different from SLM widow since display can be rescaled
map_x, map_y = 500, 500
# Directory of Zernike polynomials
polynomial_dir = "Zernike polynomials"
#number of order of parameter n of zernike polynomials to use
num_of_orders = 6
#total number of zernike terms used determined by n parameter used
num_of_terms = nmToterm(num_of_orders, num_of_orders)

'''Neural network parameters'''
input_x, input_y = 100, 100 # Size of input images for the neural network. Not the same as camera active area
train_total = 800
validation_total = 200
# factor to scale labels before training
scaling_factor = 20
label_dir = 'labels'
image_dir = 'images'
model_dir ='Net.hdf5'

# Function for measuring standard deviation of PSFs
pixel = np.empty((input_x, input_y, 2))
for i in range(input_x):
    pixel[i, :, 0] = i
for i in range(input_y):
    pixel[:, i, 1] = i
def std_measure(image):
    try:
        image = image * 255 # Rescaling image from 0-1 to 0-255
        centroid_data = np.where(image - 8 < 0, 0, image - 4) # Set pixel values less than 8 to 0 w/ other places subtract 4
        center = np.full((image.shape[0], image.shape[1], 2), (image.shape[0]//2, image.shape[1]//2))
        square = np.square(pixel - center)
        square = square.sum(axis = 2)
        return np.sqrt((square*image).sum()/image.sum())
    except:
        return 0

# Function to crop any image around the center of brightness, returns an image with dimensions suitable for neural net
def image_crop(image):
    centroid_data = np.where(image - 8 < 0, 0, image - 8)
    try:
        x, y = np.asarray(ndimage.measurements.center_of_mass(centroid_data), dtype=np.int)
    except:
        x, y = image.shape[0]//2, image.shape[1]//2
    if x < input_x//2:
        x = input_x//2
    elif x > image.shape[0] - input_x//2:
        x = image.shape[0] - input_x//2
    if y < input_y//2:
        y = input_y//2
    elif y > image.shape[1] - input_y//2:
        y = image.shape[1] - input_y//2
    return image[x - input_x//2: x - input_x//2 + input_x, y - input_y//2: y - input_y//2 + input_y]/image.max(), x, y

# Function to split images acquired on one camera into in and out of focus parts, returns the array for the neural net
def image_disect(image):
    a = image_crop(image[:, :cam_x//2])[0]
    b = image_crop(image[:, cam_x//2:])[0]
    a = np.sqrt(a * in_focus_normalization / a.sum())
    b = np.sqrt(b * out_of_focus_normalization / b.sum())
    c = np.array((a.flatten(), b.flatten()))
    return c.T.reshape(input_x, input_y, 2)

'''Function generates random Zernike coefficients satisfying atmospheric turbulence, taking the desired Fried parameter
Fried parameter tells you how strong the turbulence is.'''
S = np.loadtxt('S.csv', delimiter=',')
X = np.loadtxt('X.csv', delimiter=',')
index_table = np.loadtxt("zernike_index_table.csv",delimiter = ',').astype(np.int)
index_matrix = np.zeros((num_of_terms, num_of_terms))
for i in range(num_of_terms):
    index_matrix[index_table[i] - 1, i] = 1
def coeff_gen(fried = 1):
    B = np.random.normal(0, np.sqrt(S.diagonal()))
    A = X.dot(B)
    coefficient = index_matrix.dot(A)*np.power(fried, 5/6)
    # Remove tip and tilt
    coefficient[0: 2] = 0
    return coefficient
def show_variance(coefficients):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_yscale('log')
    ax.plot(coefficients.var(axis = 0))
    plt.show()