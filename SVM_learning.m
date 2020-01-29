%% Inizialization

addpath(genpath('test'))
addpath(genpath('train'))
TestDatasetPath  = fullfile('test');
imdsTest = imageDatastore(TestDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTrain = imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
LabelCount = countEachLabel(imdsTrain);

%% Importing Resnet18

net = resnet18;
analyzeNetwork(net);
inputSize = net.Layers(1).InputSize;

%% Rescaling and to RGB


%% Augmentation and Resizing

pixelRange = [-23 23];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandRotation',[-5,5], ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,'DataAugmentation',imageAugmenter,'ColorPreprocessing','gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize,imdsTest,'ColorPreprocessing','gray2rgb');
%% Get activators form last conv_layer

layer = 'pool5';
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

%% Get Labels

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

%% Train

classifier = fitcecoc(featuresTrain,YTrain,'Coding','onevsone');

%% Test

YPred = predict(classifier,featuresTest);

%% Results

accuracy = sum(YPred == YTest)/numel(YTest);
figure
confusionchart(YTest,YPred)