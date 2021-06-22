
clear;
close all;
%% load data
load('characters10.mat');
%   Train data
trainIdx = find(train_label==0 | train_label==8); 
trainLabel = train_label(trainIdx);
trainData = train_data(trainIdx, :);
trainLabel(trainLabel == 8) = 1;
testIdx = find(test_label == 0 | test_label == 8); 
testLabel = test_label(testIdx);
testData = test_data(testIdx, :);
testLabel(testLabel == 8) = 1; 
n_train = length(trainLabel);
n_test = length(testLabel);
n_center = 2;
stddev = 5000;
%% Cluster
center = randi(255, [n_center, size(train_data, 2)]);
cluster_label = zeros(n_train,1);
while true
    % Classify
    prev_cluster = cluster_label;
    for i = 1:n_train
        min_distance = inf;
        for j = 1:n_center
            distance = norm(double(trainData(i, :)) - center(j, :));
            if distance < min_distance
                min_distance = distance;
                cluster_label(i) = j;
            end
        end
    end
    % no update break
    if sum(prev_cluster~=cluster_label, 'all') == 0
        break
    end
    % Update
    for i = 1:n_center
        center(i, :) = mean(trainData(cluster_label==i, :), 1);
    end
end

%% RBFN
r_train = dist(trainData, uint8(center'));
RBF_train = exp(-r_train.^2/(2*stddev^2));
w = pinv(RBF_train)*trainLabel;
r_test = dist(testData, uint8(center'));
RBF_test = exp(-r_test.^2/(2*stddev^2));
pred_y_test = RBF_test * w;
pred_y_train = RBF_train * w;
loss = mse(pred_y_test, testLabel);
loss_train = mse(pred_y_train, trainLabel);

%% Performance and plot
trainAcc = zeros(1, 1000);
testAcc = zeros(1, 1000);
thr = zeros(1, 1000);
for i = 1: 1000
    t = (max(pred_y_train)-min(pred_y_train)) * (i-1)/1000 + min(pred_y_train);
    thr(i) = t;
    trainAcc(i) = (sum(trainLabel(pred_y_train<t)==0) + sum(trainLabel(pred_y_train>=t)==1)) / n_train;
    testAcc(i) = (sum(testLabel(pred_y_test<t)==0) + sum(testLabel(pred_y_test>=t)==1)) / n_test;
end
figure();
plot(thr, trainAcc, '.-', thr, testAcc, '^-');
grid on;
legend('Train','Test');
xlabel('Threshold');
ylabel('Accuracy');
fprintf('Loss train: %f\nLoss test: %f\nAcc train max: %f\nAcc test max: %f\nlen train Acc=1: %f\nlen test Acc=1: %f\nmean train accuracy: %f\nmean test accuracy: %f\n\n', ...
        loss_train, loss, max(trainAcc), max(testAcc),sum(trainAcc==1),sum(testAcc==1),mean(trainAcc), mean(testAcc));
figure();
%% cluster anlysis
% visualize center
for i = 1:n_center
    subplot(1, n_center, i);
    imshow(reshape(center(i, :), [28,28]), [0 255]);
    title(sprintf('Cluster: %d', i));
end
% compare te mean wih cluster center
figure();
mean0 = mean(trainData(trainLabel==0, :), 1);
mean1 = mean(trainData(trainLabel==1, :), 1);
subplot(1, 2, 1);
imshow(reshape(mean0, [28,28]), [0 255]);
title('Mean: 0');
subplot(1, 2, 2);
imshow(reshape(mean1, [28,28]), [0 255]);
title('Mean: 1');

figure();
residua01 = abs(mean0 - center(1, :));
residua02 = abs(mean0 - center(2, :));
residual1 = abs(mean1 - center(1, :));
residual2 = abs(mean1 - center(2, :));
subplot(2, 2, 1);
imshow(reshape(residua01, [28,28]), [0 max(residua01)]);
title('class 0 vs center 1');
subplot(2, 2, 2);
imshow(reshape(residua02, [28,28]), [0 max(residua02)]);
title('class 0 vs center 2');
subplot(2, 2, 3);
imshow(reshape(residual1, [28,28]), [0 max(residual1)]);
title('class 1 vs center 1');
subplot(2, 2,4);
imshow(reshape(residual2, [28,28]), [0 max(residual2)]);
title('class 1 vs center 2');


