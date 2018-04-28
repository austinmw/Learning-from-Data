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

# for 0,0 input
y = 0;

# cost function 
J = 0;
J = J + ((y * log(h)) + ((1 - y) * log(1 - h))) * -1;

# backpropagation
# level 3 error
delta3 = h - y;

# level 2 error
delta2 = ((THETA2' * delta3) .* (Z2 .* (1 - Z2)))(2:end);

# new weights
temp2 = THETA2 - (0.01 * (delta3 * A2'));
temp1 = THETA1 - (0.01 * (delta2 * A1'));
THETA2 = temp2;
THETA1 = temp1;

# using the new weights
Z2 = [1; THETA1 * A1];
A2 = sigmoid(Z2);
Z3 = THETA2 * A2;
h = sigmoid(Z3)

# new h should be closer to 0 (since input was 0,0)