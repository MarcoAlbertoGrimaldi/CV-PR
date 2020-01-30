%% Import

addpath(genpath('test'))
addpath(genpath('train'))

close all force
clear workspace

TestDatasetPath  = fullfile('test');
imdsTest = imageDatastore(TestDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
imds = imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
LabelCount = countEachLabel(imds);

%% Import VGG16

net = vgg16;
inputSize = net.Layers(1).InputSize;

%% Normalization


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
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');

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

% analyzeNetwork(lgraph)

%% Train

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing','gray2rgb');
options = trainingOptions('adam', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'ExecutionEnvironment','gpu', ...
    'ValidationPatience',8, ...
    'Plots','training-progress');
netTransfer = trainNetwork(augimdsTrain,lgraph,options);

%% Results

augimdsTest = augmentedImageDatastore(inputSize(1:3),imdsTest,'ColorPreprocessing','gray2rgb');

% apply the network to the test set
YPredicted = classify(netTransfer,augimdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
confusionchart(YTest,YPredicted)
