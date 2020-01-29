%% Import

addpath(genpath('test'))
addpath(genpath('train'))
close all force

imds = imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
LabelCount = countEachLabel(imds);

%% Import AlexNet

net = alexnet;
inputSize = net.Layers(1).InputSize;

%% Normalization

divideby=1;
imds.ReadFcn = @(x)double(repmat((imread(x)/divideby), 1, 1, 3));

%% Split in training and validation sets: 85% - 15%

quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

%% Augmentation and resizing

pixelRange = [-23 23];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',[-5,5], ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,'DataAugmentation',imageAugmenter);

%% Transfer

layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

% analyzeNetwork(lgraph);

%% Train

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',3, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',8, ...
    'ValidationPatience',4, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augimdsTrain,layers,options);

%% Results

TestDatasetPath  = fullfile('test');
imdsTest = imageDatastore(TestDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)double(repmat((imread(x)/divideby), 1, 1, 3));

augimdsTest = augmentedImageDatastore(inputSize,imdsTest);

% apply the network to the test set
YPredicted = classify(netTransfer,augimdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
confusionchart(YTest,YPredicted)
