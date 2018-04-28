function [parameters, costHistory] = gradient(x, y, parameters, alpha, repetition)
% main algorithm to minimize cost function
    
    % get length of data set
    m = length(y);
    
    % create a matrix of zeros for storing cost function history
    costHistory = zeros(repetition, 1);
    
    % run gradient descent
    for i = 1:repetition
        % calculate the transpose of our hypothesis
        % (theta0 + theta1*x_i - y_i  as a column-wise vector)
        % (transposed so that multiplying by row-wise vector will produce
        % scalar values for thetas)
        h = (x * parameters - y)';
        % update parameters simultaneously (check without temps later)
        temp1 = parameters(1) - alpha * (1/m) * h * x(:, 1);
        temp2 = parameters(2) - alpha * (1/m) * h * x(:, 2);
        parameters(1) = temp1;
        parameters(2) = temp2;
        % double check if this is uncessessary
        
        % keeping track of the cost function
        costHistory(i) = cost(x, y, parameters);
        
    end
    
end
