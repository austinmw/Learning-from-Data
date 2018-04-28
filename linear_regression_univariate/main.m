% http://www.codeproject.com/Articles/879043/Implementing-Gradient-Descent-to-Solve-a-Linear-Re?msg=5285341#xx5285341xx

% loading the dataset
dataSet = load('DataSet.txt');

% storing the values in separate matrices
x = dataSet(:,1); % house sqft
y = dataSet(:,3); % price
% y = dataSet(:,3) for DataSet, (:,2) for TestDataSet





% Feature normalization (helps gradient descent reach convergence)
normalization = true;
if (normalization)
    maxX = max(x);
    minX = min(x);
    x = (x - minX) / (maxX - minX);
end

% adding a column of ones to the beginning of the 'x' matrix (for theta0)
x = [ones(length(x), 1) x];

% plotting the dataset
%figure;
%plot(x(:, 2), y, 'rx', 'MarkerSize', 10);
%xlabel('Size (sqft)');
%ylabel('Price');
%title('Data Set: Housing Prices (normalized)');

% Running gradient descent on the data
% (where h_theta(x_i) = theta0 + theta1*x_i)
parameters = [0; 0]; % theta0, theta1 (y-intercept and slope)
alpha = 1; % learning rate
repetition = 100;
[parameters, costHistory] = gradient(x, y, parameters, alpha, repetition);
% use alpha = 0.01 for TestDataSet, repetition = 2000


% Plotting our cost function on a different figure to see how we did
figure;
plot(1:repetition, costHistory);
ylabel('J(theta)');
xlabel('Iterations');
title('Cost Function');


% plotting our final hypothesis
figure;
plot(min(x(:, 2)):max(x(:, 2)), parameters(1) + parameters(2) * (min(x(:, 2)):max(x(:, 2))));
xlabel('Size (sqft, normalized)');
ylabel('Price');
title('Housing Prices Linear Regression (normalized)');
hold on;
% Plotting the dataset on the same figure
plot(x(:, 2), y, 'rx', 'MarkerSize', 10);



% finally predicting the output for a provided input
% good inputs: dataset: 2200, testdataset: 5
input = 2200;
originalInput = input;
if (normalization)
        input = (input - minX) / (maxX - minX);
end
output = parameters(1) + parameters(2) * input;
plot(input, output, 'gx', 'MarkerSize', 20); 
format long g;
p = sprintf('A %d sqft. house should sell for approximately $%.2f', originalInput, output);
format long g;
disp(p);
%disp('end cost: ');
%disp(costHistory(repetition));
p2 = sprintf('theta0 is %d and theta1 is %d', parameters(1), parameters(2));
disp(p2);

