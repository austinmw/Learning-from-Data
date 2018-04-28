function [result] = sigmoid(x)
  result = 1.0 ./ (1.0 + exp(-x));
 end

# sigmoid wolframalpha: plot 1 / (1 + e^-x) from -10 to 10