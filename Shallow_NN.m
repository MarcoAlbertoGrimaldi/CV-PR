%%
%Import

close all force

imds = imageDatastore('train','IncludeSubfolders',true,'LabelSource','foldernames');
LabelCount = countEachLabel(imds);

%%
%Automatic resizing and normalization

divideby=255;
imds.ReadFcn = @(x)double(imresize(imread(x),[64 64]))/divideby;

%% 
%Example

iimage=100;
img = imds.readimage(iimage); 
figure;
imshow(img,'initialmagnification',1000)

%%
%show some instances
figure;
perm = randperm(length(imds.Labels),20);
for ii = 1:20
    subplot(4,5,ii);
    imshow(imds.Files{perm(ii)}); 
    title(imds.Labels(perm(ii)));
end
sgtitle('some instances of the training set')

%%
% split in training and validation sets: 85% - 15%
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');

%% 
% create the structure of the network
layers = [
    imageInputLayer([64 64 1],'Name','input') % 64x64 grayscale images
    
    convolution2dLayer([3 3],8,'Padding','same','Name','conv_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer([2 2],'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer([3 3],16,'Padding','same','Name','conv_2')
    reluLayer('Name','relu_2')    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    
    convolution2dLayer([3 3],32,'Padding','same','Name','conv_3')
    reluLayer('Name','relu_3')
    
    fullyConnectedLayer(15,'Name','fc_1')
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers); % to run the layers need a name
    analyzeNetwork(lgraph)
    
%% 
% training options

options = trainingOptions('sgdm', ...
    'Verbose', true, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',20, ...
    'ValidationPatience',5, ...
    'MiniBatchSize',32, ...
    'Plots','training-progress');

% train the net
net = trainNetwork(imdsTrain,layers,options);

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

