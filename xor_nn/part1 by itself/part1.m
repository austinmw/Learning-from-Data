# layer one, with bias
A1 = [1; 1; 0];

# matrix of weights from layer 1 to 2, initially random between -1 and 1
THETA1 = 2*rand(2,3) - 1;

# weights from layer 2 to 3
THETA2 = 2*rand(1,3) - 1;

# input to second layer (3x1 vector, with bias)
Z2 = [1; THETA1 * A1];

# layer 2 output before weights
A2 = sigmoid(Z2);

# input to layer 3
Z3 = THETA2 * A2;;

# final output hypothesis
h = sigmoid(Z3)


