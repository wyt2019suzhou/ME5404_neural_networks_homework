clc;
clear;
close all;
rng(520)
%% load data
load('characters10.mat');
load('neurons')
load('map_mean')
testIdx = find(test_label ~= 0 & test_label ~= 8); 
testLabel = test_label(testIdx);
testData = test_data(testIdx, :);
n_test = length(testLabel);
pred_test = zeros(size(testLabel));
M = 10;
N = 10;
correct_counter = 1;
uncorrect_counter = 1;
t= [];
%% training
for i = 1:size(testLabel,1)
    dis= squeeze(sum((double(testData(i,:))' - neurons).^2, 1))';
    [~,winner] = min(dis,[],'all','linear');
    t(i)=winner;
    row = ceil(winner/M);
    col = mod(winner, N);
    if col == 0
            col = N;
    end
    pred_test(i) = maplable(row,col);
    % plot some correct samples
    if pred_test(i,1)==testLabel(i,1) && correct_counter <= 5
        figure(1)
        sgtitle('Correct classification')
        subplot(5,2,(correct_counter-1)*2+1)
        imshow(reshape(testData(i,:),28,28))
        title(sprintf('Correct Lable:%d',testLabel(i,1)))
        subplot(5,2,(correct_counter-1)*2+2)
        imshow(reshape(neurons(:,row,col), [28 28]), [0 255],'InitialMagnification', 'fit')
        title(sprintf('Label Predicted:%d',pred_test(i,1)))
        correct_counter = correct_counter+1;
    % plot some incorrect samples
    elseif pred_test(i,1)~=testLabel(i,1) && uncorrect_counter <= 5
        figure(2)
        sgtitle('Incorrect classification')
        subplot(5,2,(uncorrect_counter-1)*2+1)
        imshow(reshape(testData(i,:),28,28))
        title(sprintf('Correct Lable:%d',testLabel(i,1)))
        subplot(5,2,(uncorrect_counter-1)*2+2)
        imshow(reshape(neurons(:,row,col), [28 28]), [0 255],'InitialMagnification', 'fit')
        title(sprintf('Label Predicted:%d',pred_test(i,1)))
        uncorrect_counter = uncorrect_counter + 1;
    end
end
accuracy = sum(pred_test == testLabel)/size(testLabel,1)