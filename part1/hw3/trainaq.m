clc;
clear;
close all;
rng(520)
%% load data
load('characters10.mat');
load('neurons')
load('map_mean')
trainIdx = find(train_label ~= 0 & train_label ~= 8); 
trainLabel = train_label(trainIdx);
trainData = train_data(trainIdx, :);
n_train = length(trainLabel);
pred_train = zeros(size(trainLabel));
M = 10;
N = 10;
correct_counter = 1;
uncorrect_counter = 1;
t= [];
%% training
for i = 1:size(trainLabel,1)
    dis = squeeze(sum((double(trainData(i,:))' - neurons).^2,1))';
    [~,winner] = min(dis,[],'all','linear');
    t(i)=winner;
    row = ceil(winner/M);
    col = mod(winner, N);
    if col == 0
            col = N;
    end
    pred_train(i) = maplable(row,col);
    % plot some correct samples
    if pred_train(i,1)==trainLabel(i,1) && correct_counter <= 5
        figure(1)
        sgtitle('Correct classification')
        subplot(5,2,(correct_counter-1)*2+1)
        imshow(reshape(trainData(i,:),28,28))
        title(sprintf('Correct Lable:%d',trainLabel(i,1)))
        subplot(5,2,(correct_counter-1)*2+2)
        imshow(reshape(neurons(:,row,col), [28 28]), [0 255],'InitialMagnification', 'fit')
        title(sprintf('Label Predicted:%d',pred_train(i,1)))
        correct_counter = correct_counter+1;
    % plot some incorrect samples
    elseif pred_train(i,1)~=trainLabel(i,1) && uncorrect_counter <= 5
        figure(2)
        sgtitle('Incorrect classification')
        subplot(5,2,(uncorrect_counter-1)*2+1)
        imshow(reshape(trainData(i,:),28,28))
        title(sprintf('Correct Lable:%d',trainLabel(i,1)))
        subplot(5,2,(uncorrect_counter-1)*2+2)
        imshow(reshape(neurons(:,row,col), [28 28]), [0 255],'InitialMagnification', 'fit')
        title(sprintf('Label Predicted:%d',pred_train(i,1)))
        uncorrect_counter = uncorrect_counter + 1;
    end
end
accuracy = sum(pred_train == trainLabel)/size(trainLabel,1)