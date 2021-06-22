clc;
clear;
close all;
rng(520)
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
%% sample
N = 100;
sampleData = datasample(trainData, N);
dmax = max(dist(sampleData), [], 'all');
stddev0 = dmax / sqrt(2*N);
r_train = dist(trainData, sampleData');
r_test = dist(testData, sampleData');
%% training
for stddev = [0.1 1 stddev0 100 1000 10000]
    RBF_train = exp(-r_train.^2/(2*stddev^2));
    w = pinv(RBF_train)*trainLabel;
    RBF_test = exp(-r_test.^2/(2*stddev^2));
    pred_y_test = RBF_test * w;
    pred_y_train = RBF_train * w;
    loss = mse(pred_y_test, testLabel);
    loss_train = mse(pred_y_train, trainLabel);
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
    title(sprintf('Stddev = %g', stddev));
    fprintf('Stddev: %g\nLoss train: %f\nLoss test: %f\nAcc train max: %f\nAcc test max: %f\nlen train Acc=1: %f\nlen test Acc=1: %f\nmean train accuracy: %f\nmean test accuracy: %f\n\n', ...
        stddev, loss_train, loss, max(trainAcc), max(testAcc),sum(trainAcc==1),sum(testAcc==1),mean(trainAcc), mean(testAcc));
end