%%
%Import

addpath(genpath('test'))
addpath(genpath('train'))
close all force

TestDatasetPath  = fullfile('test');
imdsTest = imageDatastore(TestDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
imds = imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
LabelCount = countEachLabel(imds);

%%
%Automatic resizing and normalization

divideby=255;
imds.ReadFcn = @(x)double(imresize(imread(x),[64 64]))/divideby;
imdsTest.ReadFcn = @(x)double(imresize(imread(x),[64 64]))/divideby;

%%
%Data augmentation
imageSize = [64 64];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-15,15], ...
    'RandXReflection', true, ...
    'RandXTranslation',[-8 8], ...
    'RandYTranslation',[-8 8]);

%%
% split in training and validation sets: 85% - 15%
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');
augimds_Train = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter);

%%
%show some instances
figure;
minibatch = preview(augimds_Train);
imshow(imtile(minibatch.input));
sgtitle('some instances of the augmented training set')

%% 
% create the structure of the network
layers = [
    imageInputLayer([64 64 1],'Normalization','zerocenter','Name','input') % 64x64 grayscale images
    
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
    
    convolution2dLayer([9 9],32,'Padding','same','Name','conv_4','Stride',2)
    batchNormalizationLayer('Name', 'batch_4')
    dropoutLayer( 0.25, 'Name','dropout_2')
    reluLayer('Name','relu_4')
    
    fullyConnectedLayer(256,'Name','fc_1')
    reluLayer('Name','relu_5')
    
    fullyConnectedLayer(15,'Name','fc_2')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers); % to run the layers need a name
    analyzeNetwork(lgraph)
    
%% 
% training options

options = trainingOptions('adam', ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',0.002, ...
    'Verbose', true, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',8, ...
    'ValidationPatience',8, ...
    'MiniBatchSize',64, ...
    'Plots','training-progress');

net = cell(10,1);
for i=1:10
    net{i} = trainNetwork(augimds_Train,layers,options);
end
%%
% Apply the network to the test set

YPredicted = cell(10,1);
Scores = zeros(2985,15,10);

for i=1:10
    [YPredicted{i}, Scores(:,:,i)] = classify(net{i},imdsTest);
end

%% 
% Take best match

ScoreSum = sum(Scores,3)/10;
[M,FinalYPredictedIdx] = max(ScoreSum,[],2);

T = table(YPredicted{:});
FinalYPredictedLabel = LabelCount.Label(FinalYPredictedIdx(:));

YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(FinalYPredictedLabel == YTest)/numel(YTest);

% confusion matrix
figure
confusionchart(YTest,FinalYPredictedLabel)
plotconfusion(YTest,FinalYPredictedLabel)
