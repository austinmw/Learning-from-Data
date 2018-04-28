%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---------------------------------------------------
% ENG EC 503
% Learning from Data
% Boston University
% Instructor: prakash Ishwar
% Assignment 3
% Code for part (d), Nearest Neighbor Classification
% ---------------------------------------------------
%

% edited version of Dr. Ishwar's solution with some added comments and
% possibly small changes for testing and comparing against my own results

% calculating euclidean distances by precomputing sums of squares and 
% then using inner products turns out to be a lot faster than the 
% built-in pdist2() function that I used

clear; clc;
tic

load data_mnist_train.mat
load data_mnist_test.mat
%size(X_train) % (60000x784 for mnist X_train)

%  uncomment for memory efficiency (increases runtime)
% X_train = sparse(X_train);
% X_test = sparse(X_test);


%
% Precompute the sum of squares term for speed
%
XtrainSOS = sum(X_train.^2,2);  %(ends up obviously at 60000x1 for mnist)
%XtestSOS = sum(X_test.^2,2);

ntest = length(Y_test);
nbatches = 20;
% batches is nbatches number of row vectors each with ntest/nbatches
% number of elements. The elements are integers 1:ntest split evenly
% into batches{1} to batches{nbatches}
batches = mat2cell(1:ntest,1,(ntest/nbatches)*ones(1,nbatches));
% transpose SOS to col vec and replicate ntest/nbatches times => 500x60000
X_temp = repmat(XtrainSOS', ntest/nbatches,1);

Y_pred = zeros(ntest,1);
%
% Classify test points
%
for i=1:nbatches
    i %#ok<*NOPTS>
    % computing squared euclidean distance via inner products trick (knn slide #18) or:
    % http://nonconditional.com/2014/04/on-the-trick-for-computing-the-squared-euclidian-distances-between-two-sets-of-vectors/
    dst = -2*X_test(batches{i},:)*X_train' + X_temp; % (500x60000)
    % min distances for batch
    [junk,closest] = min(dst,[],2);  % (closest: 500x1, junk: 500x1)
    % batch labeling based on label of closest training point
    Y_pred(batches{i}) = Y_train(closest);
end

%
% Report results
%
errorRate = mean(Y_pred ~= Y_test);
CCR = 1 - errorRate
CFmtx = confusionmat(Y_pred, Y_test);

toc
