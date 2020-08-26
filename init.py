import numpy as np

from library import num_of_orders, num_of_terms, nmToterm, map_x, map_y, polynomial_dir

#initializing host polynomial memory array
h_polynomial = np.zeros((num_of_terms, map_x, map_y), np.float32)
#initializing host addition result array
h_result = np.zeros((map_x, map_y), np.float32)
for n in range(1, num_of_orders + 1):
    for m in range(-n , n + 2, 2):
        term = nmToterm(n, m) - 1
        filename = polynomial_dir + '\\n={}_m={}.npy'.format(n, m)
        h_polynomial[term] = np.load(filename)

def add(coefficients):
    global h_result
    h_result = np.zeros((map_x, map_y), np.float32)
    for j in range(0, num_of_terms):
        h_result = h_result + coefficients[j] * h_polynomial[j]
    h_result += 0.15
    h_result = h_result % 1
    return h_result