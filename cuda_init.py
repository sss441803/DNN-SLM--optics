from numba import cuda, int32, float32, void
import numpy as np
from string import Template
import time

from library import num_of_orders, num_of_terms, nmToterm, map_x, map_y, polynomial_dir, blockSize

#add function for cuda kernel to execute
@cuda.jit#(void(int32, int32, int32, float32, float32, float32, float32))
def cuda_add(num_of_terms, coefficients, d_polynomial, d_result):
    #index of individual threads in each block
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    #number of threads working in parallel
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(index, map_x * map_y, stride):
        for j in range(0, num_of_terms):
            m = i % map_x
            n = i // map_x
            d_result[n][m] = (d_result[n][m] + coefficients[j] * d_polynomial[j * map_x * map_y + i])
        # + 0.15 so that 0 phase is not at the center, which avoids having white-black transition in the middle
        d_result[n][m] += 0.15
        # Wrapping the phase
        d_result[n][m] = d_result[n][m] % 1

#initializing host polynomial memory array
h_polynomial = np.zeros((num_of_terms, map_x, map_y), np.float32)
#initializing host addition result array
h_result = np.zeros((map_x, map_y), np.float32)

#importing polynomials
for n in range(1, num_of_orders + 1):
    for m in range(-n , n + 2, 2):
        term = nmToterm(n, m) - 1
        filename = polynomial_dir + '\\n={}_m={}.npy'.format(n, m)
        h_polynomial[term] = np.load(filename)

h_polynomial = h_polynomial.ravel()
#copying polynomials from host to device
d_polynomial = cuda.device_array(num_of_terms * map_x * map_y, dtype=np.float32)
d_polynomial = cuda.to_device(h_polynomial)
#releasing device memory from polynomials imported
del h_polynomial
#creating device memory for addition result
d_result = cuda.to_device(h_result)

#calculating total number of blocks needed for one addition operation
numBlocks = (map_x * map_y + blockSize - 1) // blockSize

def add(coefficients):
    global h_result
    h_result = np.zeros((map_x, map_y), np.float32)
    d_result = cuda.to_device(np.zeros((map_x, map_y), dtype=np.float32))
    cuda.synchronize()
    cuda_add[numBlocks, blockSize](num_of_terms, coefficients, d_polynomial, d_result)
    h_result = d_result.copy_to_host()
    return h_result