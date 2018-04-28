% Austin Welch
% EC503 HW4

% parts a,b,c,d,e (not f)

% clear old data and start timing script
clear; clc;
tic

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
numWordsTrain = zeros(nDocsTrain,1);
for i=1:nDocsTrain    
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

% (b)
fprintf('(b)\n\n');

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



% comment out while testing following section

% How many of the W x 20 estimated parameters that are zero
nZeroBetas = sum(sum(likelihoods==0));
fprintf('Number of the W x 20 estimated parameters that equal zero: ');
fprintf('%d\n\n', nZeroBetas);
%fprintf('Percentage of Betas that equal zero: ');
%fprintf('%0.2f%%\n\n', 100*nZeroBetas/numel(likelihoods));

% probabilities for each document and class
probabilities = zeros(length(Y_test),nClasses); % 7505 x 20
counts = histc(X_test(:,1),unique(X_test(:,1)));
docs = mat2cell(X_test,counts);
for n=1:length(Y_test) % 7505 iterations
    result = bsxfun(@times, docs{n}(:,3), log(likelihoods(docs{n}(:,2),:)));
    % result ends up size length(doc) x 20   
    probabilities(n,:) = sum(result);
end


% add priors to betas
probabilities = bsxfun(@plus, probabilities, logpriors');

% Number of test docs where all class probabilities are zero
noinf = probabilities;
noinf(noinf==-inf) = 0;
allzerodocs = 0;
for i=1:length(noinf)
   if sum(noinf(i,:)) == 0
     allzerodocs = allzerodocs+1;
   end
end
fprintf('Number of test docs where all class probabilities are zero: ');
fprintf('%d\n\n', allzerodocs);

% CCR result
[~,predictions] = max(probabilities,[],2);
CCR = sum(predictions==Y_test)/length(Y_test);
fprintf('Test CCR (part b): %0.4f\n\n', CCR);

% (c)
fprintf('(c)\n\n');

% Remove all words from test that aren't in train
X_test = X_test(~ismember(X_test(:,2), testExclusiveWords), :);

% redo same calculations with modified X_test

% probabilities for each document and class
probabilities = zeros(length(Y_test),nClasses); % 7505 x 20


counts = histc(X_test(:,1),unique(X_test(:,1)));
docs = mat2cell(X_test,counts);
for n=1:length(Y_test) % 7505 iterations
    result = bsxfun(@times, docs{n}(:,3), log(likelihoods(docs{n}(:,2),:)));
    % result ends up size length(doc) x 20   
    probabilities(n,:) = sum(result);
end

% Total number of non-zero betas
nnonzbetas = sum(sum(probabilities~=-inf));
fprintf('Total number of non-zero betas: %d\n\n', nnonzbetas);

% add priors to betas
probabilities = bsxfun(@plus, probabilities, logpriors');

% Number of test docs where all class probabilities are zero
noinf = probabilities;
noinf(noinf==-inf) = 0;
allzerodocs = 0;
for i=1:length(noinf)
   if sum(noinf(i,:)) == 0
     allzerodocs = allzerodocs+1;
   end
end

%fprintf('Number of test docs where all class probabilities are zero: ');
%fprintf('%d\n\n', allzerodocs);

% CCR result
[~,predictions] = max(probabilities,[],2);
CCR = sum(predictions==Y_test)/length(Y_test);
fprintf('Test CCR (part c): %0.4f\n\n', CCR);


% (d)
fprintf('(d)\n\n');

% Dirichlet prior on betas (+ 1/W)
likelihoods = likelihoods + (1/nwords_all);

% redo calculations again

% probabilities for each document and class
probabilities = zeros(length(Y_test),nClasses); % 7505 x 20
counts = histc(X_test(:,1),unique(X_test(:,1)));
docs = mat2cell(X_test,counts);
for n=1:length(Y_test) % 7505 iterations
    result = bsxfun(@times, docs{n}(:,3), log(likelihoods(docs{n}(:,2),:)));
    % result ends up size length(doc) x 20   
    probabilities(n,:) = sum(result);
end

% add priors to betas
probabilities = bsxfun(@plus, probabilities, logpriors');

% CCR result
[~,predictions] = max(probabilities,[],2);
CCR = sum(predictions==Y_test)/length(Y_test);
fprintf('Test CCR (part d): %0.4f\n\n', CCR);

% confusion matrix
fprintf('\t\t\t\t   Confusion matrix: prediction totals for each class\n');
cmat = confusionmat(Y_test,predictions);
disp(cmat);
figure(1);
imshow(cmat, [], 'InitialMagnification',5000);  
colormap(jet);
title('Heatmap for confusion matrix', 'FontSize', 15);

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

figure(2);
plot(log(dirichlets),ccrs);
title('log(a-1) vs. CCR');
xlabel('log(a-1)');
ylabel('CCR');




% (f)
%fprintf('(f)\n\n');

% slow than other parts, not left out of this script


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\n');
toc


