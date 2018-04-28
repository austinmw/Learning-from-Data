% Austin Welch
% EC503 HW4_1a

clear; clc;

% load data
X_train = importdata('train.data'); % doc_ID, word_ID, word_count
Y_train = importdata('train.label');
X_test = importdata('test.data');
Y_test = importdata('test.label');
newsGroups = importdata('newsgrouplabels.txt');
vocab = importdata('vocabulary.txt');

% (a) 
fprintf('(a)\n\n');

% unique words
train_words = unique(X_train(:,2));
nwords_train = length(train_words);
test_words = unique(X_test(:,2));
nwords_test = length(test_words);
concat_words = [train_words',test_words'];
all_words = unique(concat_words);
nwords_all = length(all_words);
fprintf('unique words in train set: %d\n', nwords_train);
fprintf('unique words in test set: %d\n', nwords_test);
fprintf('unique words entire dataset: %d\n\n', nwords_all);

% average document length

% training set
[rowsTrain, ~] = size(X_train);
nDocsTrain = length(Y_train); % length(unique(X_train(:,1))) == nDocsTrain;
%fprintf('Number of training docs: %d\n\n', nDocsTrain);
[~, docStartIndexTrain, ~] = unique(X_train(:,1));
rangeTrain = nDocsTrain;
numWordsTrain = zeros(rangeTrain,1);
for i=1:rangeTrain    
    s = docStartIndexTrain(i);   
    if i < nDocsTrain 
        f = docStartIndexTrain(i+1)-1;
    else
        f = docStartIndexTrain(i:end);
    end
    numWordsTrain(i) = sum(X_train(s:f,3));    
end
meanWordsTrain = mean(numWordsTrain);
fprintf('Mean number of words in training docs: %0.2f\n', meanWordsTrain);

% test set
[rowsTest, ~] = size(X_test);
nDocsTest = length(Y_test); % length(unique(X_test(:,1))) == nDocsTest;
[~, docStartIndexTest, ~] = unique(X_test(:,1));
numWordsTest = zeros(nDocsTest,1);
for i=1:nDocsTest    
    s = docStartIndexTest(i);   
    if i < nDocsTest 
        f = docStartIndexTest(i+1)-1;
    else
        f = docStartIndexTest(i:end);
    end
    numWordsTest(i) = sum(X_test(s:f,3));    
end
meanWordsTest = mean(numWordsTest);
fprintf('Mean number of words in test docs: %0.2f\n\n', meanWordsTest);

% number of words unique words in test that are not in train
testExclusiveWords = setdiff(test_words, train_words);
nTestExclusive = length(testExclusiveWords);
fprintf('Number of unique words in test that are not in train: ');
fprintf('%d\n\n', nTestExclusive);

