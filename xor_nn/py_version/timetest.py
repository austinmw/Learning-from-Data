import time
import xor_nn
import numpy as np

XOR = np.array([[0,0,0], [0,1,1], [1,0,1], [1,1,0]])

THETA1, THETA2 = xor_nn.xor_nn(XOR, 0, 0, 1, 1, 0.01)

t_start = time.clock()
for i in range(10000):
	THETA1, THETA2 = xor_nn.xor_nn(XOR, THETA1, THETA2, 0, 1, 0.01)

t_end = time.clock()
print('Elapsed time ', t_end - t_start)