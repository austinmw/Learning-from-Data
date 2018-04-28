%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [W, test_acc, train_acc, test_logloss,obj]=logistic_reg_l2(y, X, test_Y, test_X, lambda, c)
% ENG EC 503: Learning from Data
% Boston University,
% Instructor: Prakash Ishwar
% Assignment 5, Problem 5.1
% Supporting function for part (b):
% Logistic Regression with L2 norm regularization for classification
% Inputs:
% y: training labels
% X: training data
% test_Y: test label
% test_X: test data
% lambda: L2 regularizaion parameter
% c: step-size, chose to be fixed
% the number of iterations is set to 1000 by default
%
% Output:
% test_acc: test accuracies in CCR for each iteration (a 1000 * 1 vector)
% train_acc: training accuracies in CCR for each iteration (a 1000 * 1
% vector)
% test_logloss: test accuracies in Logloss for each iteration (a 1000 * 1
% vector)
% obj: objective function value for each iteration (a 1000 * 1 vector)
% W: dim * K matrix, each column is the weight vector for a class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% some default setting if not specified
    if ~exist('lambda', 'var')
        lambda=1000;
    end
    if ~exist('c', 'var')
        c=1e-5;
    end
    % set up maximum iteration as stopping criteria
    maxiter=1000;
    K=length(unique(y)); % number of classes
    p=size(X,2); % dimension of feature
    W=zeros(p,K);
    % saving traing/test accuracy and objective values for each iteration
    test_acc=zeros(maxiter,1);
    train_acc=zeros(maxiter,1);
    test_logloss = zeros(maxiter,1);
    obj=zeros(maxiter,1);
    % start gradient descent algorithm
    iter=1;
    while (iter <= maxiter)
        % Calculate gradient (for w's) in matrix form

        G=log_grad(y,X,W)-lambda*W;
        W=W+c*G;
        % Compute objective function value, the trainning/test accuracy at the current iteration
        obj(iter)=log_obj(y, X, W)-lambda/2*sum(W(:).^2);
        train_acc(iter)=cal_te_acc(W, y, X);
        test_acc(iter)=cal_te_acc(W, test_Y, test_X);
        test_logloss(iter)=cal_te_logloss(W,test_Y, test_X);
        % Print itermediate result
        if mod(iter,10)==0
            fprintf('Iter=%d, Obj=%f, tr_acc=%f, te_acc=%f\n,te_logloss=%f\n', ...
                iter, obj(iter), train_acc(iter), test_acc(iter), ...
                test_logloss(iter));
        end
        iter=iter+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function G=log_grad(y, X, W)
% ENG EC 503: Learning from Data
% Boston University,
% Instructor: Prakash Ishwar
% Assignment 5, Problem 5.1
% Supporting function for part (b):
% function to calculate gradients (for w's) in logistic regression
% (NLL only, without regularization terms)
% Input:
% y - labels (N * 1 vector)
% X - data (N * Dim data vector), each row represent a data point
% W - the current weight parameters
% - Dim * K-1 matrix, each column is the weight vector w for one class
% - (from class 1 up to class K-1)
% Output:
% The gradient of the maximum likelihood estimator for logsitic regression
% G: Dim * K matrix (same dimension as W), each column is the gradient
% vector d_W for one class
    K=size(W,2);
    eXB=exp(X*W);
    s_eXB=1./(sum(eXB, 2)+1);
    eXB=eXB.*repmat(s_eXB, 1, K);
    for k= 1: K
        eXB(:,k)=(y==k)-eXB(:,k);
    end
    G=(X'*eXB);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function obj=log_obj(y, X, W)
% ENG EC 503: Learning from Data
% Boston University,
% Instructor: Prakash Ishwar
% Assignment 5, Problem 5.1
% Supporting function for part (b):
% Calculate objective function value
% Input:
% y : (labels) N * 1 vector, numberical values from 1 to K, where K is the
% number of classes
% X : (data) N * dim matrix, each row is a data points. dim is the dimension
% of the features
% W - the current weight parameters
%- Dim * K matrix, each column is the weight vector w for one class
%
% Output:
% obj : the objective value / Negative Log-Likelihood
    K=size(W,2);
    n=length(y);
    XW=X*W; % inner products (n * K)
    obj_1=log(sum(exp(XW), 2));
    idx=sub2ind([n,K], [1:n]', y);
    obj_2=zeros(n,1);
    obj_2=XW(idx); % was written as obj_2(I), but think this was an error
    obj=sum(obj_2-obj_1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function acc=cal_te_acc(W, te_label, te_data)
% ENG EC 503: Learning from Data
% Boston University,
% Instructor: Prakash Ishwar
% Assignment 5, Problem 5.1
% Supporting function for part (b):
% Calculate the accuracy of the logistic regression predictor
% The accuracy is measured using Correct classification rates (CCR)
% assuming K classes. the labels are numbers between 1 and K
% Input:
% W - the weight parameters
% - Dim * K matrix, each column is the weight vector w for one class
% te_label - the test labels
% te_data - the test data points
% Output:
% acc - CCR
    n_te=size(te_data,1);
    w=te_data*W;
    [~, pred_label]=max(w,[],2);
    acc=1-sum(pred_label~=te_label)/n_te;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function logloss=cal_te_logloss(W, te_label, te_data)
% ENG EC 503: Learning from Data
% Boston University,
% Instructor: Prakash Ishwar
% Assignment 5, Problem 5.1
% Supporting function for part (b):
% Calculate the accuracy of the logistic regression predictor
% The accuracy is measured using log loss
% assuming K classes. the labels are numbers between 1 and K
% Input:
% W - the weight parameters
% - Dim * K-1 matrix, each column is the weight vector w for one class
% - (from class 1 up to class K-1)
% te_label - the test labels
% te_data - the test data points
    n_te=size(te_data,1);
    w=te_data*W;
    % calculate log-loss
    Idx = sub2ind(size(w),[1:n_te]', te_label);
    obj1 = exp(w(Idx));
    obj2 = sum(exp(w),2);
    logloss = sum(log(obj2./obj1))/n_te;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%