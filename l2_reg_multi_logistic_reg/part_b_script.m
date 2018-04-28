%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 503: Learning from Data
% Boston University,
% Instructor: Prakash Ishwar
% Assignment 5, Problem 5.1
% Script for Part (b) Regularized Multi-class Logistic Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load features
if ~exist('Features','var')
    load data_SFcrime_train.mat;
    binarizeSFdata;
end
% Get the training/test split
[N,D]=size(Features);
N_train = ceil(0.6*N);
N_test = N - N_train;
X_train = Features(1:N_train,:);
Y_train = Crimes_Y(1:N_train);
X_test = Features(N_train+1:end, :);
Y_test = Crimes_Y(N_train+1:end, :);
% Train Logistic Regression classifier with L2 regularization
lambda = 1000; % regularization param.
c=1e-5; % gradient descent step size
[W_0, te_err_0, tr_err_0, te_logloss, obj_0]=logistic_reg_l2(Y_train, ...
X_train, Y_test, X_test, lambda, c);
% Generate all the plots
figure;
plot([1:1000], -obj_0,'k-');
xlabel('number of iteration'), ylabel('objective values')
figure;
plot([1:1000],te_err_0,'k-');
xlabel('number of iteration'), ylabel('test CCR')
figure;
plot([1:1000],te_logloss,'k-');
xlabel('number of iteration'),ylabel('test Logloss')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
