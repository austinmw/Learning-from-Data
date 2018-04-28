% Austin Welch
% EC503 HW4_1e

clear; clc;
tic

% load data
X_train = importdata('train.data'); % doc_ID, word_ID, word_count
Y_train = importdata('train.label');
X_test = importdata('test.data');
Y_test = importdata('test.label');

% unique words
train_words = unique(X_train(:,2));
nwords_train = length(train_words);
test_words = unique(X_test(:,2));
nwords_test = length(test_words);
concat_words = [train_words',test_words'];
all_words = unique(concat_words);
nwords_all = length(all_words);

% training set
[rowsTrain, ~] = size(X_train);
nDocsTrain = length(Y_train); % length(unique(X_train(:,1))) == nDocsTrain;
[~, docStartIndexTrain, ~] = unique(X_train(:,1));
% test set
[rowsTest, ~] = size(X_test);
nDocsTest = length(Y_test); % length(unique(X_test(:,1))) == nDocsTest;
[~, docStartIndexTest, ~] = unique(X_test(:,1));

% number of words unique words in test that are not in train
testExclusiveWords = setdiff(test_words, train_words);
nTestExclusive = length(testExclusiveWords);

% classes and number of classes
classes = unique(Y_train);
nClasses = length(classes);

% counts fof each class in training data
classCounts = zeros(nClasses,1);
for i=1:nClasses   
    classCounts(i) = length(find(Y_train==i)); 
end

% prior for each class
priors = classCounts / length(Y_train); % used for MAP
logpriors = log(priors);

% determine number of rows (words) for each label from number of documents
% and counts for each label
lengths = zeros(nClasses,1);
last = 0; index = 1;
for i=1:nClasses   
    if i==20
        lengths(i) = length(X_train(last+1:end,:));
    else    
        index = index + classCounts(i);
        lastWord = docStartIndexTrain(index)-1;
        lengths(i) = lastWord - last;
        last = lastWord;
    end    
end

% group training data by class into matrices within cell array
groupedXtrain = mat2cell(X_train, lengths);

% sum of every word for each class
totalClassWords = zeros(nClasses, 1);
for i=1:nClasses
    totalClassWords(i) = sum(groupedXtrain{i}(:,3));
end

% individual word totals for each class
classEachWordCount = zeros(nwords_all, numel(groupedXtrain));
for k = 1:numel(groupedXtrain)
    classEachWordCount(:,k) = accumarray(groupedXtrain{k}(:,2), ...
        groupedXtrain{k}(:,3), [nwords_all 1], @sum);
end

% probabilities for each word in each class (betas)
likelihoods = bsxfun(@rdivide, classEachWordCount, totalClassWords');


% (e)
fprintf('(e)\n\n');

% Dirichlet prior on betas (+ 1/W)
dirichlets = [10^-5, 10^-4.5, 10^-4, 10^-3.5, 10^-3, 10^-2.5, ...
    10^-2, 10^-1.5, 10^-1, 10^-0.5, 10^0, 10^0.5, 10^1, 10^1.5];
ccrs = length(dirichlets);

for i=1:length(dirichlets)
    
    D = dirichlets(i);
    
    fprintf('(a-1) = %0.6f, ', D);
    likelihoodsW = likelihoods + D;

    % Remove all words from test that aren't in train
    X_test = X_test(~ismember(X_test(:,2), testExclusiveWords), :);

    % redo same calculations with modified X_test

    % probabilities for each document and class
    probabilities = zeros(length(Y_test),nClasses); % 7505 x 20
    counts = histc(X_test(:,1),unique(X_test(:,1)));
    docs = mat2cell(X_test,counts);
    for n=1:length(Y_test) % 7505 iterations
        result = bsxfun(@times, docs{n}(:,3), log(likelihoodsW(docs{n}(:,2),:)));
        % result ends up size length(doc) x 20   
        probabilities(n,:) = sum(result);
    end

    % add priors to betas
    probabilities = bsxfun(@plus, probabilities, logpriors');

    % CCR result
    [~,predictions] = max(probabilities,[],2);
    CCR = sum(predictions==Y_test)/length(Y_test);
    fprintf('     Test CCR: %0.4f\n\n', CCR);
    
    ccrs(i) = CCR;
end

plot(log(dirichlets),ccrs);
title('log(a-1) vs. CCR');
xlabel('log(a-1)');
ylabel('CCR');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\n');
toc

