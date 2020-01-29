%% Import

addpath(genpath('test'))
addpath(genpath('train'))
close all force
clear workspac

imds = imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
LabelCount = countEachLabel(imds);

%% Import AlexNet

net = alexnet;
inputSize = net.Layers(1).InputSize;

%% Normalization

mean = 0;
variance = 1;

imds.ReadFcn = @(x)double(repmat(((imread(x)-mean)/variance), 1, 1, 3));

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

    for i = 1:numel(layersTransfer)
        if isprop(layersTransfer(i),'WeightLearnRateFactor')
            layersTransfer(i).WeightLearnRateFactor = 0;
        end
        if isprop(layersTransfer(i),'WeightL2Factor')
            layersTransfer(i).WeightL2Factor = 0;
        end
        if isprop(layersTransfer(i),'BiasLearnRateFactor')
            layersTransfer(i).BiasLearnRateFactor = 0;
        end
        if isprop(layersTransfer(i),'BiasL2Factor')
            layersTransfer(i).BiasL2Factor = 0;
        end
    end
    
    
numClasses = numel(categories(imdsTrain.Labels));

lgraph = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
    softmaxLayer
    classificationLayer];

% analyzeNetwork(lgraph);

%% Train

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',8, ...
    'ValidationPatience',4, ...
    'Verbose',false, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augimdsTrain,lgraph,options);

%% Results

TestDatasetPath  = fullfile('test');
imdsTest = imageDatastore(TestDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)double(repmat(((imread(x)-mean)/variance), 1, 1, 3));
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

% apply the network to the test set
YPredicted = classify(netTransfer,augimdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
confusionchart(YTest,YPredicted)
