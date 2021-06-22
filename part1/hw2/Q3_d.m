
clear;
clc;
%% Input and resize Data
groupID = 3;
train_dir = dir('Face Database/TrainImages/*.jpg');
test_dir = dir('Face Database/TestImages/*.jpg');
label_train_dir = dir('Face Database/TrainImages/*.att');
label_test_dir = dir('Face Database/TestImages/*.att');
num_train = size(label_train_dir, 1);
num_test = size(label_test_dir, 1);
img_train = zeros(10201, num_train);
img_test = zeros(10201, num_test);
label_train = zeros(1, num_train);
label_test = zeros(1, num_test);

for i = 1:num_train
    img_name = train_dir(i).name;
    label_name = label_train_dir(i).name;
    img = imread(['Face Database/TrainImages/', img_name]);
    img = rgb2gray(img);
    img_v = img(:);
    img_train(:, i) = img_v(1:10201);
    label = load(['Face Database/TrainImages/',  label_name]);
    label_train(i) = label(groupID);
end
for j = 1:num_test
    img_name = test_dir(j).name;
    label_name = label_test_dir(j).name;
    img = imread(['Face Database/TestImages/', img_name]);
    img = rgb2gray(img);
    img_v = img(:);
    img_test(:, j) = img_v(1:10201);
    label = load(['Face Database/TestImages/', label_name]);
    label_test(j) = label(groupID);
end

all_img=[img_train ,img_test];
all_label=[label_train,label_test];

%% multi-layer batch train
n_hidden=50;%100
net = patternnet(n_hidden,'traincgb');
% net = patternnet(n_hidden);
net = configure(net,all_img,all_label);
net.performFcn = 'mse';
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 6;
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:size(img_train,2)*0.8;
net.divideParam.valInd = size(img_train,2)*0.8+1:size(img_train,2);
net.divideParam.testInd= size(img_train,2)+1:size(all_img,2);
net.trainParam.lr=0.0001;
net.trainParam.goal=0.0001;
[net, tr]  = train(net, all_img, all_label);

%% anlysis
pred_train = net(img_train(:,1:size(img_train,2)*0.8));
pred_val = net(img_train(:,size(img_train,2)*0.8+1:size(img_train,2)));
pred_test = net(img_test);
perf_train = 1 - mean(abs(label_train(:,1:size(img_train,2)*0.8)-pred_train));
perf_vaild =1-mean(abs(label_train(:,size(img_train,2)*0.8+1:size(img_train,2))-pred_val));
perf_test = 1 - mean(abs(label_test-pred_test));
disp(perf_train)
disp(perf_vaild)
disp(perf_test)
