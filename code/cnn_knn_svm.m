%% ASL Alphabet Recognition - Multi-Algorithm Comparison
% Comparing KNN, CNN, and SVM approaches

%% Step 1: Load and Prepare Dataset
datasetPath = 'E:\asltrain\asl'; % Replace with your path
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display dataset info
disp(['Total images: ', num2str(numel(imds.Files))]);
labelCounts = countEachLabel(imds);
disp('Class distribution:');
disp(labelCounts);
%% 
bar(labelCounts.Count); % Horizontal bars
xticks(1:29);
xticklabels(labelCounts.Label);
title('Number of Samples per ASL Letter');
%% Step 2: Data Splitting (60% Train, 20% Val, 20% Test)
[trainData, tempData] = splitEachLabel(imds, 0.6, 'randomized');
[valData, testData] = splitEachLabel(tempData, 0.5, 'randomized');


%% Step 3: Common Preprocessing
% Function to resize and normalize images
preprocess = @(img) im2double(imresize(rgb2gray(imread(img)), [64 64]));

% Process all sets
trainFeatures = zeros(numel(trainData.Files), 64*64);
valFeatures = zeros(numel(valData.Files), 64*64);
testFeatures = zeros(numel(testData.Files), 64*64);
%%
for i = 1:numel(trainData.Files)
    disp(i);
    trainFeatures(i,:) = reshape(preprocess(trainData.Files{i}), 1, []);
end
%% 52200 
for i = 1:numel(valData.Files)
    disp(i);
    valFeatures(i,:) = reshape(preprocess(valData.Files{i}), 1, []);
end

%%
for i = 1:numel(testData.Files)
    disp(i);
    testFeatures(i,:) = reshape(preprocess(testData.Files{i}), 1, []);
end

trainLabels = trainData.Labels;
valLabels = valData.Labels;
testLabels = testData.Labels;


%%
subplot(1,2,1); imshow(imread(trainData.Files{1})); title('Original');  
subplot(1,2,2); imshow(reshape(trainFeatures(1,:), [64 64])); title('Processed'); 
%% ========== KNN Implementation ==========
disp('----- Training KNN Classifier -----');

% Find optimal k
kValues = 1:2:15;
knnAccuracies = zeros(size(kValues));

for i = 1:length(kValues)
    disp(i);
    knnModel = fitcknn(trainFeatures, trainLabels, ...
        'NumNeighbors', kValues(i), ...
        'Standardize', true);
    
    valPred = predict(knnModel, valFeatures);
    knnAccuracies(i) = sum(valPred == valLabels)/numel(valLabels);
end

% Plot k vs accuracy
figure;
plot(kValues, knnAccuracies, '-o');
xlabel('k value');
ylabel('Validation Accuracy');
title('KNN Performance vs k Value');
grid on;

% Train final model with best k
[bestAcc, bestIdx] = max(knnAccuracies);
bestK = kValues(bestIdx);
finalKNN = fitcknn([trainFeatures; valFeatures], [trainLabels; valLabels], ...
    'NumNeighbors', bestK, ...
    'Standardize', true);

% Evaluate
knnPred = predict(finalKNN, testFeatures);
knnAccuracy = sum(knnPred == testLabels)/numel(testLabels);
disp(['KNN Test Accuracy: ', num2str(knnAccuracy*100), '%']);

%%
subplot(1,2,1); imshow(imread(trainData.Files{1})); title('Original');  
subplot(1,2,2); imshow(reshape(trainFeatures(1,:), [64 64])); title('Processed');  
%% 
confusionchart(testLabels, knnPred);  
title('KNN (k=1) Confusion Matrix');  

%% 
% Plot k vs accuracy
figure;
plot(kValues, knnAccuracies, '-o');
xlabel('k value');
ylabel('Validation Accuracy');
title('KNN Performance vs k Value');
grid on;

%% ========== SVM Implementation ==========
disp('----- Training SVM Classifier -----');

% Train SVM with RBF kernel
svmModel = fitcecoc(trainFeatures, trainLabels, ...
    'Learners', templateSVM('KernelFunction', 'rbf', ...
    'BoxConstraint', 1, ...
    'KernelScale', 'auto'));

% Validate
valPred = predict(svmModel, valFeatures);
svmValAcc = sum(valPred == valLabels)/numel(valLabels);

% Train final on combined train+val
finalSVM = fitcecoc([trainFeatures; valFeatures], [trainLabels; valLabels], ...
    'Learners', templateSVM('KernelFunction', 'rbf', ...
    'BoxConstraint', 1, ...
    'KernelScale', 'auto'));

% Evaluate
svmPred = predict(finalSVM, testFeatures);
svmAccuracy = sum(svmPred == testLabels)/numel(testLabels);
disp(['SVM Test Accuracy: ', num2str(svmAccuracy*100), '%']);

%% ========== CNN Implementation ==========
disp('----- Training CNN Classifier -----');

% Create augmented datastores (no augmentation transforms)
inputSize = [64 64 3];
trainAug = augmentedImageDatastore(inputSize, trainData, 'ColorPreprocessing', 'gray2rgb');
valAug = augmentedImageDatastore(inputSize, valData, 'ColorPreprocessing', 'gray2rgb');
testAug = augmentedImageDatastore(inputSize, testData, 'ColorPreprocessing', 'gray2rgb');

% Define CNN architecture
layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(256)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(numel(categories(trainData.Labels)))
    softmaxLayer
    classificationLayer
];

% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 64, ...
    'ValidationData', valAug, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train CNN
[cnnNet, info] = trainNetwork(trainAug, layers, options);

% Evaluate
cnnPred = classify(cnnNet, testAug);
cnnAccuracy = sum(cnnPred == testLabels)/numel(testLabels);
disp(['CNN Test Accuracy: ', num2str(cnnAccuracy*100), '%']);

%% ========== Comparison and Visualization ==========
% Accuracy comparison
algorithms = {'KNN'; 'SVM'; 'CNN'};
accuracies = [knnAccuracy; svmAccuracy; cnnAccuracy];

figure;
bar(accuracies);
set(gca, 'XTickLabel', algorithms);
ylabel('Accuracy');
title('Algorithm Comparison on ASL Recognition');
ylim([0 1]);
grid on;

% Add accuracy values on bars
for i = 1:numel(accuracies)
    text(i, accuracies(i)+0.02, sprintf('%.2f%%', accuracies(i)*100), ...
        'HorizontalAlignment', 'center');
end

% Confusion matrices
figure;
subplot(1,3,1);
confusionchart(testLabels, knnPred);
title('KNN Confusion Matrix');

subplot(1,3,2);
confusionchart(testLabels, svmPred);
title('SVM Confusion Matrix');

subplot(1,3,3);
confusionchart(testLabels, cnnPred);
title('CNN Confusion Matrix');

%% Save Results
save('asl_results.mat', 'finalKNN', 'finalSVM', 'cnnNet', ...
    'knnAccuracy', 'svmAccuracy', 'cnnAccuracy');
disp('All models and results saved to asl_results.mat');