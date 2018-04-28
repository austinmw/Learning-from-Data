XOR = [0,0,0; 0,1,1; 1,0,1; 1,1,0]; # 3 is y, not bias, bias is moved to A1, A2

THETA1 = 0;
THETA2 = 0;

# has init_w = 1 in order to initialize the weights
[THETA1, THETA2] = xor_nn(XOR, THETA1, THETA2, 1, 1, 1.0);

# now init_w is 0, since already initialized
for i = 1:2000
  [THETA1, THETA2] = xor_nn(XOR, THETA1, THETA2, 0, 1, 1.0); 

  if (mod(i,500) == 0)
    disp('Iteration : '), disp(i)
    [THETA1, THETA2] = xor_nn(XOR, THETA1, THETA2);
  endif  
  
endfor


# can run a lot faster with alpha=1.0, i=1:2000, mod(i,500)