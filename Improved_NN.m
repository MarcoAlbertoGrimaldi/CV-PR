%%
%Import

addpath(genpath('test'))
addpath(genpath('train'))
close all force

imds = imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
LabelCount = countEachLabel(imds);

%% Normalization

divideby=255;
imds.ReadFcn = @(x)double(imresize(imread(x),[64 64]))/divideby;

%%
%Data augmentation
imageSize = [64 64];
imageAugmenter = imageDataAugmenter( ...
    'RandXShear',[-6,6], ...
    'RandYShear',[-6,6], ...
    'RandRotation',[-10,10], ...
    'RandXReflection', true, ...
    'RandXTranslation',[-6 6], ...
    'RandYTranslation',[-6 6]);

%%
% split in training and validation sets: 85% - 15%
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');
imdsTrain.ReadFcn  = @(x)double(imresize(imread(x),[64 64]))/divideby;
imdsValidation.ReadFcn = @(x)double(imresize(imread(x),[64 64]))/divideby;
augimds_Train = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);
augimds_Validation = augmentedImageDatastore(imageSize,imdsValidation);

%%
% %show some instances
% figure;
% minibatch = preview(augimds_Train);
% imshow(imtile(minibatch.input));
% sgtitle('some instances of the augmented training set')

%% 
% create the structure of the network
layers = [
    imageInputLayer([64 64 1],'Name','input') % 64x64 grayscale images
    
    convolution2dLayer([3 3],8,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name', 'batch_1')
    reluLayer('Name','relu_1')
    
    convolution2dLayer([5 5],16,'Padding','same','Name','conv_2','Stride',1)
    batchNormalizationLayer('Name', 'batch_2')
    dropoutLayer( 0.25, 'Name','dropout_1')
    reluLayer('Name','relu_2')    
    
    maxPooling2dLayer(2,'Stride',3,'Name','maxpool_1')
    
    convolution2dLayer([7 7],32,'Padding','same','Name','conv_3','Stride',2)
    batchNormalizationLayer('Name', 'batch_3')
    reluLayer('Name','relu_3')
    
    convolution2dLayer([9 9],64,'Padding','same','Name','conv_4','Stride',2)
    batchNormalizationLayer('Name', 'batch_4')
    dropoutLayer( 0.25, 'Name','dropout_2')
    reluLayer('Name','relu_4')
   
    
    fullyConnectedLayer(15,'Name','fc_2')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers); % to run the layers need a name
%     analyzeNetwork(lgraph)
    
%% 
% training options

options = trainingOptions('adam', ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',0.002, ...
    'Verbose', true, ...
    'ValidationData',augimds_Validation, ...
    'ValidationFrequency',8, ...
    'ValidationPatience',6, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');

% train the net
net = trainNetwork(augimds_Train,layers,options);

%%

TestDatasetPath  = fullfile('test');
imdsTest = imageDatastore(TestDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)double(imresize(imread(x),[64 64]))/divideby;

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
confusionchart(YTest,YPredicted)

